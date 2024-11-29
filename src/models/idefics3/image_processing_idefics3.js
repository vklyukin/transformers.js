

import {
    ImageProcessor,
} from "../../base/image_processors_utils.js";
import { cat, full, interpolate_4d } from "../../utils/tensor.js";

export class Idefics3ImageProcessor extends ImageProcessor {
    constructor(config) {
        super(config);

        this.do_image_splitting = config.do_image_splitting ?? true;
        this.max_image_size = config.max_image_size;
    }

    /**
     * Calculate size to resize images to, to be multiples of `vision_encoder_max_size` while preserving the aspect ratio.
     * @param {import('../../utils/tensor.js').Tensor} pixel_values Tensor of the image to resize.
     * @param {number} vision_encoder_max_size Maximum size of the output image. If the image is larger than this size,
     * it will be split into patches of this size, and the original image will be concatenated with the patches, resized to max_size.
     */
    get_resize_for_vision_encoder(pixel_values, vision_encoder_max_size) {
        let [height, width] = pixel_values.dims.slice(-2);

        const aspect_ratio = width / height;
        if (width >= height) {
            width = Math.ceil(width / vision_encoder_max_size) * vision_encoder_max_size;
            height = Math.floor(width / aspect_ratio);
            height = Math.ceil(height / vision_encoder_max_size) * vision_encoder_max_size;
        } else {
            height = Math.ceil(height / vision_encoder_max_size) * vision_encoder_max_size;
            width = Math.floor(height * aspect_ratio);
            width = Math.ceil(width / vision_encoder_max_size) * vision_encoder_max_size;
        }
        return { height, width };
    }

    // /** @param {RawImage|RawImage[]|RawImage[][]} images */
    async _call(images, {
        do_image_splitting = null,
        return_row_col_info = false,
    } = {}) {
        // TODO: support 2D RawImages
        if (!Array.isArray(images)) {
            images = [images];
        }

        let images_list = await Promise.all(images.map(x => this.preprocess(x)));

        // Original sizes of images
        const original_sizes = images_list.map(x => x.original_size);

        // Reshaped sizes of images, before padding or cropping
        const reshaped_input_sizes = images_list.map(x => x.reshaped_input_size);

        // Convert images to 4D tensors for easier processing
        images_list.forEach(x => x.pixel_values.unsqueeze_(0));

        let pixel_values;
        let images_list_rows = [];
        let images_list_cols = [];

        const { longest_edge } = this.max_image_size;

        if (do_image_splitting ?? this.do_image_splitting) {
            let image_rows = new Array(images_list.length);
            let image_cols = new Array(images_list.length);

            // We first resize both height and width of each image to the nearest max_image_size multiple, disregarding the aspect ratio
            images_list = await Promise.all(images_list.map(async (x, i) => {
                const new_size = this.get_resize_for_vision_encoder(x.pixel_values, longest_edge);

                const resized = await interpolate_4d(x.pixel_values, {
                    size: [new_size.height, new_size.width],
                });

                const { frames, num_splits_h, num_splits_w } = await this.split_image(resized, this.max_image_size);
                image_rows[i] = num_splits_h;
                image_cols[i] = num_splits_w;
                return cat(frames, 0);
            }));

            images_list_rows.push(image_rows);
            images_list_cols.push(image_cols);
        } else {
            /** @type {[number, number]} */
            const size = [longest_edge, longest_edge];
            images_list = await Promise.all(
                images_list.map(x => interpolate_4d(x.pixel_values, { size }))
            );

            images_list_rows.push(new Array(images_list.length).fill(0));
            images_list_cols.push(new Array(images_list.length).fill(0));
        }

        // Stack pixel values
        // TODO: support 2D images inputs
        pixel_values = cat(images_list, 0);
        pixel_values.unsqueeze_(0);

        // TODO: Improve pixel_attention_mask
        const [b, n, c, h, w] = pixel_values.dims;
        const pixel_attention_mask = full([b, n, h, w], true);

        return {
            pixel_values,
            pixel_attention_mask,

            original_sizes,
            reshaped_input_sizes,
            ...(
                return_row_col_info
                    ? { rows: images_list_rows, cols: images_list_cols }
                    : {}
            ),
        }
    }

    async split_image(pixel_values, { longest_edge }) {
        const max_height = longest_edge;
        const max_width = longest_edge;

        const frames = [];

        const [height, width] = pixel_values.dims.slice(-2);

        let num_splits_h = 0, num_splits_w = 0;

        if (height > max_height || width > max_width) {
            // Calculate the number of splits
            num_splits_h = Math.ceil(height / max_height);
            num_splits_w = Math.ceil(width / max_width);

            // Calculate the optimal width and height for the sub-images
            const optimal_height = Math.ceil(height / num_splits_h);
            const optimal_width = Math.ceil(width / num_splits_w);

            // Iterate through each row and column
            for (let r = 0; r < num_splits_h; r++) {
                for (let c = 0; c < num_splits_w; c++) {
                    // Calculate the starting point of the crop
                    const start_x = c * optimal_width;
                    const start_y = r * optimal_height;

                    // Calculate the ending point of the crop
                    const end_x = Math.min(start_x + optimal_width, width);
                    const end_y = Math.min(start_y + optimal_height, height);

                    // Crop the image
                    frames.push(pixel_values.slice(null, null, [start_y, end_y], [start_x, end_x]));
                }
            }

            // Resize the global image to match max dimensions for memory efficiency
            const global_image_height = max_height;
            const global_image_width = max_width;

            if (height !== global_image_height || width !== global_image_width) {
                pixel_values = await interpolate_4d(pixel_values, {
                    size: [global_image_height, global_image_width],
                })
            }
        }

        frames.push(pixel_values);

        return { frames, num_splits_h, num_splits_w };
    }
}

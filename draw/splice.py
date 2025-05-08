from PIL import Image, ImageDraw, ImageFont

def concatenate_images(image_paths, output_path, direction='horizontal', target_size=None):
    """
    Concatenates images into one, resizing them to the same width and height if necessary.

    Args:
        image_paths (list): List of image file paths to concatenate.
        output_path (str): File path to save the concatenated image.
        direction (str): Direction of concatenation, 'horizontal' or 'vertical'.
        target_size (tuple): Desired width and height (width, height) for all images. If None, use minimum size of input images.
    """
    try:
        # Load images
        images = [Image.open(path) for path in image_paths]

        # Determine target width and height
        if target_size is None:
            target_width = min(img.width for img in images)
            target_height = min(img.height for img in images)
        else:
            target_width, target_height = target_size

        # Resize all images to the target size
        images = [img.resize((target_width, target_height)) for img in images]

        # Calculate the total size of the final image
        if direction == 'horizontal':
            total_width = sum(img.width for img in images)
            max_height = target_height
            result = Image.new("RGB", (total_width, max_height))
            
            # Paste images side by side
            x_offset = 0
            for img in images:
                result.paste(img, (x_offset, 0))
                x_offset += img.width
        elif direction == 'vertical':
            max_width = target_width
            total_height = sum(img.height for img in images)
            result = Image.new("RGB", (max_width, total_height))
            
            # Paste images one below the other
            y_offset = 0
            for img in images:
                result.paste(img, (0, y_offset))
                y_offset += img.height
        else:
            raise ValueError("Direction must be 'horizontal' or 'vertical'.")

        # Save the resulting image
        result.save(output_path)
        print(f"Concatenated image saved to {output_path}")

    except Exception as e:
        print(f"Error occurred: {e}")


# Example usage:
# Paths to the images
# image_paths = ["/mnt/hy/data/splice/qualitative experiment/rome/gt.png", "/mnt/hy/data/splice/qualitative experiment/rome/3dgs.png",\
#                 "/mnt/hy/data/splice/qualitative experiment/rome/ours.png","/mnt/hy/data/splice/qualitative experiment/rome/ours_large.png"]
# output_path = "/mnt/hy/data/splice/spliced/rome.jpg"

# image_paths = ["/mnt/hy/data/splice/spliced/00000.jpg", "/mnt/hy/data/splice/spliced/00001.jpg",\
#                 "/mnt/hy/data/splice/spliced/00009.jpg", "/mnt/hy/data/splice/spliced/00013.jpg"]
# image_paths = ["/mnt/hy/data/splice/spliced/00014.jpg", "/mnt/hy/data/splice/spliced/00016.jpg",\
#                 "/mnt/hy/data/splice/spliced/00021.jpg", "/mnt/hy/data/splice/spliced/00028.jpg",\
#                 "/mnt/hy/data/splice/spliced/00032.jpg"]
image_paths = ["/mnt/hy/data/splice/spliced/bilbao.jpg", "/mnt/hy/data/splice/spliced/hollywood.jpg",\
                "/mnt/hy/data/splice/spliced/pompidou.jpg", "/mnt/hy/data/splice/spliced/quebec.jpg",\
                "/mnt/hy/data/splice/spliced/rome.jpg"]
# image_paths = ["/mnt/hy/data/splice/spliced/00003.jpg", "/mnt/hy/data/splice/spliced/00012.jpg"]
# image_paths = ["/mnt/hy/data/splice/spliced/00006.jpg", "/mnt/hy/data/splice/spliced/00030.jpg"]
# image_paths = ["/mnt/hy/data/splice/spliced/00008.jpg", "/mnt/hy/data/splice/spliced/00024.jpg",\
#                 "/mnt/hy/data/splice/spliced/00038.jpg", "/mnt/hy/data/splice/spliced/00048.jpg",\
#                 "/mnt/hy/data/splice/spliced/00067.jpg"]
output_path = "/mnt/hy/data/splice/spliced/bungeenerf-2.jpg"

# Concatenate images horizontally
# concatenate_images(image_paths, output_path, direction='horizontal')
concatenate_images(image_paths, output_path, direction='vertical')


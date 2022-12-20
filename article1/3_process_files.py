import os
from wand import image
from rembg import remove
from PIL import Image, UnidentifiedImageError

def has_transparency(img):
    # img is expected to be a PIL Image
    if isinstance(img, image.Image): # wand image format
        return img.alpha_channel
    else:
        if img.info.get("transparency", None) is not None:
            return True
        elif img.mode == "P":
            transparent = img.info.get("transparency", -1)
            indices = [idx == transparent for _, idx in img.getcolors()]
            return all(indices)
        elif img.mode == "RGBA":
            extrema = img.getextrema()
            return extrema[3][0] < 255
        else:
            return False

def process_files():
    '''
    1. Loop through all files and folders
    2. If a tiff file exists without a PNG copy, open it
    3. If there's 4 channels, no need to remove bg
    4. But if there is only 3, then remove bg
    5. If a tiff file exists with a PNG copy, open it
    6. If the PNG file has 3 channels, remove bg, otherwise ignore it
    '''

    # some files caused fatal errors so I added them here
    # to this list so that execution could proceed somewhat unhindered
    badfiles = []
    for root, dirs, files in os.walk(base):
        for f in files:
            if (f.endswith(".tif") or f.endswith(".tiff")) and f not in badfiles:
                print(f"Checking file {f}")
                tiffpath = os.path.join(root, f)
                pngpath = tiffpath.replace(".tiff", ".png")
                pngpath = pngpath.replace(".tif", ".png")
                try:
                    if os.path.exists(pngpath):
                        img = Image.open(pngpath)
                        transparent = has_transparency(img)
                    else:
                        img = Image.open(tiffpath)
                        transparent = has_transparency(img)
                    # set height to 1000 px and calculate new width
                    new_size = (int(1000 / img.size[1] * img.size[0]), 1000)
                    if not transparent:
                        print("Removing background and resizing")
                        output = remove(img)
                        # resize image
                        resized = output.resize(new_size) # w, h
                        resized.save(pngpath)
                        print("Background removed and resized")
                    else:
                        if img.size[1] != 1000:
                            print("Resizing image")
                            resized = img.resize(new_size) # w, h
                            resized.save(pngpath)
                            print("Image resized")
                except UnidentifiedImageError:
                    # ImageMagick can read 5-channel TIFFs
                    img = image.Image(filename=tiffpath)
                    transparent = has_transparency(img)

                    # all 5-channel TIFFs were transparent
                    if transparent:
                        w, h = img.size # width, height
                        if h != 1000:
                            img.resize(int(w / h * 1000), 1000)
                            img.save(filename=pngpath)
if __name__ == '__main__':
    base = os.getcwd()
    os.chdir(base)
    process_files()

import matplotlib.pyplot as plt
plt.ion()

img_dir = "/home/jev/sfb/images/"

fig_file_dict = {"A":"lmmtfr_fig1_SO_no2,3,28,14,51_group_rsyncfact_0.tif",
                 "B":"ERPAC_SO_fix_wavelet.png",
                 "C":"ND_SO_wavelet_12-15_250-650_predict.png",
                 "D":"polar_hist_fig1_SO_wavelet.png"}

mosaic_str = """
             AAAAAAAAAA
             AAAAAAAAAA
             AAAAAAAAAA
             FBBBBCCCCG
             FBBBBCCCCG
             DDDDDDDDDD
             DDDDDDDDDD
             DDDDDDDDDD
             """

fig, axes = plt.subplot_mosaic(mosaic_str, figsize=(18,38.4))

for k,v in fig_file_dict.items():
    pass
    axes[k].axis("off")
    axes["F"].axis("off")
    axes["G"].axis("off")
    im = plt.imread(img_dir+v)
    axes[k].imshow(im, aspect="equal")
    axes[k].text(0, 0.9, k, transform=axes[k].transAxes, fontsize=38)
plt.tight_layout()
plt.savefig("{}fig2.png".format(img_dir))

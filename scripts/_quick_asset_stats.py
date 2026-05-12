import glob, os, numpy as np, cv2, ncnn

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
assets_dir = os.path.join(workspace_root, "app", "src", "main", "assets")
param = os.path.join(assets_dir, "yolo26n_safehat.ncnn.param")
binf = os.path.join(assets_dir, "yolo26n_safehat.ncnn.bin")
if not (os.path.exists(param) and os.path.exists(binf)):
    param = os.path.join(assets_dir, "yolo26n_e2e.ncnn.param")
    binf = os.path.join(assets_dir, "yolo26n_e2e.ncnn.bin")
imgs = sorted(glob.glob(os.path.join(workspace_root, "data", "valid", "images", "*")))[:40]
net = ncnn.Net(); net.load_param(param); net.load_model(binf)
allmax=[]
for p in imgs:
    bgr = cv2.imread(p)
    if bgr is None: continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h0,w0 = rgb.shape[:2]
    target=640
    if w0>h0:
        scale=target/float(w0); w=target; h=int(h0*scale)
    else:
        scale=target/float(h0); h=target; w=int(w0*scale)
    m = ncnn.Mat.from_pixels_resize(rgb, ncnn.Mat.PixelType.PIXEL_RGB, w0, h0, w, h)
    wpad=((w+31)//32*32-w); hpad=((h+31)//32*32-h)
    mpad = ncnn.copy_make_border(m, hpad//2, hpad-hpad//2, wpad//2, wpad-wpad//2, ncnn.BorderType.BORDER_CONSTANT, 114.0)
    mpad.substract_mean_normalize([], [1/255.0,1/255.0,1/255.0])
    ex = net.create_extractor(); ex.input("in0", mpad)
    _, out = ex.extract("out0")
    out_np = np.array(out)
    if out_np.ndim == 3: out_np = out_np[0]
    # Expect (14,8400)
    if out_np.shape[0] < out_np.shape[1]:
        cls = out_np[4:,:]
        mx = cls.max(axis=0)
    else:
        cls = out_np[:,4:]
        mx = cls.max(axis=1)
    allmax.append(mx)
if not allmax:
    print('no data')
    raise SystemExit(1)
arr=np.concatenate(allmax)
print('samples',arr.size,'p50',float(np.quantile(arr,0.5)),'p90',float(np.quantile(arr,0.9)),'p99',float(np.quantile(arr,0.99)),'gt090',float((arr>0.9).mean()),'gt099',float((arr>0.99).mean()))

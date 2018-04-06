#Pyfaster RCNN Implementation

### Disclaimer

The official Faster R-CNN code (written in MATLAB) is available [here](https://github.com/ShaoqingRen/faster_rcnn).
If your goal is to reproduce the results in our NIPS 2015 paper, please use the [official code](https://github.com/ShaoqingRen/faster_rcnn).

This repository contains a Python *reimplementation* of the MATLAB code.
This Python implementation is built on a fork of [Fast R-CNN](https://github.com/rbgirshick/fast-rcnn) for multitiff frames, aimed to accelerating the training of faster R-CNN object detection models. Recently, there are a number of good implementations:

* https://github.com/sulth/faster-rcnn-resnet, developed based on Pycaffe + Numpy.
* https://github.com/xinleipan/py-faster-rcnn-with-new-dataset.

### Hardware 

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |



To train this faster rcnn model on a new dataset, follow the instructions below. 

#### Step 1: Setup Faster-RCNN:

follow https://gist.github.com/vaibhawchandel/f829f5b8aff9b55a6473ccaf5f8db4bd for the installation insructions.

#### Step 2: Download models and run a demo

    cd ..
    ./data/scripts/fetch_faster_rcnn_models.sh
    ./data/scripts/fetch_imagenet_models.sh

run demo.py "python ./tools/demo.py --gpu 0 --net vgg16"

Make sure you've successfully finished all parts mentioned above without error message. Then you can proceed.

#### Step 4: Prepare your own dataset

This is the most important step, and you should modify multiple files inside $Faster-RCNN-Root/lib.

(1) Transform your data into the format of VOC2007. The pascal voc 2007 dataset are composed of three folders: ImageSets, JPEGImages, Annotations. You should build your own dataset like this. The annotations file should be ".xml" format. Create the dataset using the script "tools/create_xml.py". Then you divide your dataset into trainval, val, train, test part. Then you delete the original "VOC2007" folder inside the "VOCdevkit" folder with the new dataset folder you just created. 

(2) change the file "$Faster-RCNN-Root/lib/datasets/pascal_voc.py" line 30 "self._classes" to the actuall classes of your datasets

Since you are using a different dataset, the number of classes might be different and the format of your picture might also be different from ".jpg", so you may also need to change "self._image_ext = '.jpg'" accordingly. 

(3)ZF Training steps

1)./tools/train_net.py --gpu 0 --solver models/pascal_voc/ZF/faster_rcnn_end2end/solver.prototxt --weights data/faster_rcnn_models/ZF_faster_rcnn_final.caffemodel --imdb voc_2007_trainval --iters 250000 --cfg experiments/cfgs/faster_rcnn_end2end.yml --iter 0

2)nohup ./tools/train_net.py --gpu 0 --solver models/pascal_voc/ZF/faster_rcnn_end2end/solver.prototxt --weights output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_iter_0.caffemodel --imdb voc_2007_trainval --iters 250000 --cfg experiments/cfgs/faster_rcnn_end2end.yml --iter 50000 &

3)./tools/test_net.py --gpu 0 --def models/pascal_voc/ZF/faster_rcnn_end2end/test.prototxt --net output/faster_rcnn_end2end/voc_2007_trainval/zf_faster_rcnn_iter_10000.caffemodel --imdb voc_2007_trainval --cfg experiments/cfgs/faster_rcnn_end2end.yml 

We have trained and tested on VGG16 and Resnet-101 model as well.


### Issues and solutions :

(1)i tried to resolve error with numpy version as :
a)version 1.13—>index not an int
b)downgraded it to 1.11 —>unable to import multi array.
Solution :
 You can also fix the problem by modifying lib/roi_data_layer/minibatch.py

On line 26, there's a call to np.round
fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)
This, unfortunately, creates a float, which causes the issues. Changing this line to
fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image).astype(np.int)
will fix this without having to use an older version of numpy, since fg_rois_per_image is where the floats were creeping in and causing trouble.

I believe there are only 3 other places where floats are being used to index, so you'd have to add
.astype(np.int) to the end of

lib/datasets/ds_utils.py line 12 : hashes = np.round(boxes * scale).dot(v)
lib/fast_rcnn/test.py line 129 : hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
lib/rpn/proposal_target_layer.py line 60 : fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

I guess if you're really bored you can sniff out any remaining problematic float indexing attempts, or if you're not then maybe yes, you should install numpy 1.11.0 as suggested above!

(2)how detect multiple object on same image.

   call following block of code on main()

im = im[:, :, (2, 1, 0)]
fig, ax = plt.subplots(figsize=(12, 12))
 ax.imshow(im, aspect='equal')


for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets =  dets[keep, :]
        vis_detections(ax, cls, dets,  thresh=CONF_THRESH)
def vis_detections(ax, class_name , dets, thresh=0.5):    
"""Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    print(inds)

    if len(inds) == 0:
          return
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        print(bbox[0],bbox[1],bbox[2],bbox[3])
        ax.add_patch(
                     plt.Rectangle((bbox[0], bbox[1]),
                     bbox[2] - bbox[0],
                     bbox[3] - bbox[1], fill=False,
                     edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,  thresh), fontsize=14)
    plt.draw()
    
    i hope this works for you !
    
    
(3)This tools/demo.py works for the multitiff file processing.



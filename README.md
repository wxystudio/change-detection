# change detection

 
 This is a Change Detection project I did during my internship at [INNNO](http://www.innno.net/#/)

 公司需求：公司的业务包括用无人机来检测某一地区不同时间可能会发生的变化，包括且不限于：房屋的变化，河流的改道，树林的增殖等等，但是用人工来判断两张图片的变化费时费力，我们就需要无人机拍摄好图片，用人工智能的手段来检测某地区的地形是否发生了变化。
 
 前期处理：由于无人机拍摄过程中的抖动和角度等原因，同一路线飞过的图片不可能完全对齐，所以我们先对图片进行了对齐处理，尽量使得同一路线不同时间拍摄的两张图片，四个角是对齐的。
 
 具体过程：在训练过程中，我们将有变化的两张图片按照channel合并起来，再将有变化的部分标注出mask，放入模型中训练。训练完成后，在同一地区不同时间拍摄的两组照片，将其按照channel叠加起来，放入训练好的模型中，可以检测出地区内的房屋，水利等等的变化情况，具体会由mask显示出来。

 需要检测的图片存放于fakeA中，检测的结果位于resultA中，模型已经训练好。由于数据的缺乏，我们并不是真的拿两个时间的照片进行训练，而是将某些房屋抹掉，人为营造出了变化。
 左边是需要检测出房屋的照片，我们之后用工具抹掉了房屋作为对比图。右边是检测出的结果，可以发现我们的房屋被检测出来了

Company requirements: Using drones to detect changes that may occur in a certain area at different times, including but not limited to: changes in houses, rivers and forests, etc. This work is time-consuming and labor-intensive. We need drones to take high-quality pictures and use artificial intelligence to detect whether the terrain of a certain area has changed.

Pre-processing: Due to the jitter and angle during the shooting process of the drone, the pictures taken in the same route cannot be completely aligned, so we first align the pictures, and try to make two pictures taken at different times on the same route. The four corners are aligned.

Specific process: During the training process, we merge the two pictures that have changed according to the channel, and then mark the changed part as a mask and put it into the model for training. After the training is completed, two sets of photos taken at different times in the same area are superimposed according to channels and put into the trained model to detect changes in houses, water conservancy, etc. in the area, which will be displayed by the mask. come out.

The pictures to be detected are stored in fakeA, the detection results are in resultA, and the model has been trained. Due to the lack of data, we didn't actually train on photos of two times, but erased some houses and artificially created changes. On the left is the photo where the house needs to be detected. We then erased the house with a tool as a comparison image. On the right is the detected result, we can find that our house has been detected
 
 ![这是需要检测出房屋的照片，我们之后用工具抹掉了房屋作为对比图](https://github.com/wxystudio/change-detection/blob/master/fake_A/1000.jpg)
 ![这是检测出的结果，可以发现我们的房屋被检测出来了](https://github.com/wxystudio/change-detection/blob/master/resultA/1000.jpg)

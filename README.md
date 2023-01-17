# Objects_edges_detection_wide_screen

To Study objects edges detection in wide screens, there are variables for object detection in wide screen without masksing or colours techniques.

### Objectives ###

1. Automatic edges and scales detection in wide backgrounds area.
2. Easy implement, integrations and understanding with simple steps can be transfroms.

### Problems ###

1. Without masking do not compared to objects we know or target shape or dimension but only edges detection from elapsed of padding zero screen input and original image ( the same as velocity continuos action image can locates of the continuous movement but we do it on single image with padding zero )
2. There are varieties of background, contrastivness, brightness and locations are not specific from random input image result variables input created.
3. As the most image object locators they required loop or multiple of neurons layer with min-max or regularizations, multiple of convolution layers saved more time than grids but time taken to build.
4. For certainty results we use initializer for kernel and bias, Tensorflow use strides that different from camera of devices they are using edges detection matrixes as in cannon or modren filters camera but strides do the same when you applied the correct strides number or tuple.

#### Object locator wide screen background ####

See background output is min-max from few convilution networks. the bigger model we are using same as probided on the Internet but this one I experiments by myself to understand how dies it working as previous projects, see GIF image.

![Figure 1](https://github.com/jkaewprateep/Objects_edges_detection_wide_screen/blob/main/Figure_1.png "Figure 1")

#### Resizes image ####

![Figure 5](https://github.com/jkaewprateep/Objects_edges_detection_wide_screen/blob/main/Figure_5.png "Figure 5")

#### Convolution layers, image edges detection ####

![Figure 6](https://github.com/jkaewprateep/Objects_edges_detection_wide_screen/blob/main/Figure_6.png "Figure 6")


```
FILE = "F:\\Pictures\\actor-Ploy\\005.jpg"
image = tf.io.read_file( FILE )
image = tf.io.decode_jpeg( image )
image = o_image = o_image_2 = tf.keras.utils.img_to_array(image)

o_image = tf.image.resize( o_image, [ 96, 96 ] )
o_image = tf.expand_dims(o_image, 0)
conv_image_2 = tf.keras.layers.Conv2D( 3, (2, 2), kernel_initializer=tf.ones_initializer(), 
      bias_initializer=tf.ones_initializer(), strides=(1, 1), activation='linear', padding="same")( o_image )

paddings = tf.constant([[0, 0], [3, 2], [3, 2], [0, 0]])
conv_image_3 = tf.pad(conv_image_2, paddings, "CONSTANT")
conv_image_3 = tf.image.resize( conv_image_3, [ 96, 96 ] )

conv_image_2 = tf.image.resize( conv_image_2, [ 96, 96 ] )
conv_image_2 = conv_image_3 - conv_image_2

plt.imshow( tf.keras.utils.array_to_img( tf.squeeze( conv_image_2 ) ) )
plt.show()
```

| File name | Description |
| --- | --- |
| Figure_1.png | image scan regions |
| Figure_5.png | resizes image |
| Figure_6.png | edges enchangement image |
| Figure_7.png | result 1 |
| Figure_8.png | result 2 |
| Figure_10.png | result 3 |
| bandicam 2022-12-17%2011-31-54-783.gif | 301 |
| README.md | readme file |

## Result ##

#### Cell density and distance ####

![Figure 7](https://github.com/jkaewprateep/Objects_edges_detection_wide_screen/blob/main/Figure_7.png "Figure 7")

#### Multiple objects edges randomness ####

![Figure 8](https://github.com/jkaewprateep/Objects_edges_detection_wide_screen/blob/main/Figure_8.png "Figure 8")

#### Target object seekings ####

![Figure 10](https://github.com/jkaewprateep/Objects_edges_detection_wide_screen/blob/main/Figure_10.png "Figure 10")

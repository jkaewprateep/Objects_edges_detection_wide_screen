# Objects_edges_detection_wide_screen
To Study objects edges detection in wide screens.

#### Object locator wide screen background ####

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

## Result ##

#### Cell density and distance ####

![Figure 7](https://github.com/jkaewprateep/Objects_edges_detection_wide_screen/blob/main/Figure_7.png "Figure 7")

#### Multiple objects edges randomness ####

![Figure 8](https://github.com/jkaewprateep/Objects_edges_detection_wide_screen/blob/main/Figure_8.png "Figure 8")

#### Target object seekings ####

![Figure 10](https://github.com/jkaewprateep/Objects_edges_detection_wide_screen/blob/main/Figure_10.png "Figure 10")

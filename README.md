Fake Face Replace Space in Case of Chase we Race to Erase then Place, incorporated
aka FFRSCCREP, inc.

To run the program:

`main.py [path/to/img.jpg]`

For example:

`main.py TestImages/group.jpg`

If you do not provide an argument path to an image, it will open the default image path defined on line 20 as defaultPath.
If you provide an argument but it is not a valid path to an image, the program will display an error then exit.

After the program opens the input image, it will run facial detection and draw a green square over each deteced face.
Clicking in any of these rectangles will turn them blue. This will then become the selected face.
After selecting a face, pressing G will generate a fake face and place it over the selected face.
You can press G as many times as you want until you get a face you like.

After a face has been generated, it will not be saved until you press the S key. 
This will save the image in its normal path but with -modified added to the file name.
This will also clear the selected face.

After generating a face, but before saving it, you also have the following options:
* B: Increase brightness
* V: Decrease brightness
* C: Increase contrast
* X: Decrease contrast
* R: Return generated face to before any modifications made
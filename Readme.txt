The goal of the project was to implement an optical character recognition system. The input to the code is various characters, as an output the code should be able to detect and recognize those characters from text.
The project was carried out in three stages described below.

Enrollment
- First, I binarized all the character images from character list. Then, I extracted characters from image and stored in a dictionary. I multiplied each character pixel by 255 so that it can be detected by canny edge detector.
- Using canny edge detector and setting lower and higher threshold 100 and 200 respectively, I extracted edges of characters and store those features(edges) into a dictionary. The return function of enrollment is a dictionary with key as character name and value as their features.

Detection
- First, I binarized the test image. Then using connected component labeling, I labeled all the characters in the test image. Then I created bounding box around all labeled characters to specify their location in test image. I stored bounding box with respective label in a dictionary.
- I extracted all the labels from the labeled image and stored them in a dictionary as per English text reading method. I set pixel values of all labels 255 so that it can be detected by canny edge detector.
- The detection function returns dictionary of detected labels and a dictionary of bounding box.

Recognition
- For each label detected in detection part, I extracted edges using canny edge detector. Then I compared features of the label and all enrolled characters by resizing enrolled character features.
- I performed matching using sum of squared difference (SSD). By analyzing minimum error (SSD) of recognized characters I set threshold to 0.27.
- Using evaluate.py I calculated F1 score which is 0.8.

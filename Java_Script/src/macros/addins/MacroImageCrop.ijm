File.openSequence("D:/FRI/ING/DP/GIT_REPO/Java_Script/resources/Samples/tmp/");
if (false) {
	waitForUser("Create Region of Interest Selection");
	run("ROI Manager...");
	roiManager("Add");
	roiManager("List");
	saveAs("Results", "D:/FRI/ING/DP/GIT_REPO/Java_Script/resources/Outputs/cropped//region.csv");
} else {
	makeRectangle(324, 1254, 2646, 1656);
}
run("Crop");
run("Image Sequence... ", "select=D:/FRI/ING/DP/GIT_REPO/Java_Script/resources/Outputs/cropped/ dir=D:/FRI/ING/DP/GIT_REPO/Java_Script/resources/Outputs/cropped/ format=JPG use");
close();
run("Quit");

imgnm  = "20181211_175204";
title = imgnm + ".jpg";
title1 = imgnm + "-1.jpg";
open("D:/FRI/ING/DP/GIT_REPO/Java_Script/resources/Samples/" + title);
run("Duplicate...", " ");
getStatistics(mean, std);
if((mean > 165 && std > 76) || std >= 98) {
	setMinAndMax(110, 158);
} else if ((mean < 150 && std < 90) 
	|| (mean > isNeutral && std < 98)) {
	setMinAndMax(150, 4);
} else { setMinAndMax(30, 150);}
run("Split Channels");
selectWindow("" + title1 + " (green)");
run("Enhance Contrast...", "saturated=0.4 normalize equalize");
run("Auto Threshold", "method=Minimim ");
run("Convert to Mask");
//for (i=0; i<4; i++) {run("Dilate");}
run("Dilate");
run("Analyze Particles...", "size=8000-103000 circularity=0.10-1.00 show=Outlines display exclude clear summarize overlay add");
selectWindow(title);
roiManager("Show All");
run("Flatten");
saveAs("Png", "D:/FRI/ING/DP/GIT_REPO/Java_Script/resources/Outputs/stats//img/" + imgnm + ".png");
saveAs("Results", "D:/FRI/ING/DP/GIT_REPO/Java_Script/resources/Outputs/stats//results/Results_" + imgnm + ".csv");
roiManager("List");
saveAs("Results", "D:/FRI/ING/DP/GIT_REPO/Java_Script/resources/Outputs/stats//list/List_" + imgnm + ".csv");
run("Quit");

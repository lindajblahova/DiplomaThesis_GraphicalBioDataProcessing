package code.evaluation;

import code.common.Common;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import org.json.JSONObject;
import org.json.JSONTokener;

import java.awt.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;

public class Stats {

    private ArrayList<LinkedList<Integer[]>> outlinesAll;
    private ArrayList<LinkedList<Integer[]>> labeledAll;
    private String pathOutlines;
    private String pathLabels;
    private List<String> fileNameList;
    private ArrayList<Integer[]> matches; // [[find ok] [more] [less]]
    private String datasetName;
    final String statsOutputDirParamName = "statsOutputDirParam";
    final String thresholdDistanceCellPxParamName = "thresholdDistanceCellPxParam";
    final String thresholdDistanceCellPxParamValue = "300";

    /**
     * read outlines in one file CSV with name
     *
     * @param fileNameOutline
     */
    private void readOutlines(String fileNameOutline) {

        LinkedList<Integer[]> outlinesImage = new LinkedList<>();
        String csvFile;
        String splitBy = "\n";
        String delimiter = ",";
        String pathOutline = pathOutlines + "list\\List_" + fileNameOutline + ".csv";

        try {

            int a = 0;
            csvFile = new Scanner(new File(pathOutline)).useDelimiter("\\Z").next();
            String[] lines = csvFile.split(splitBy);

            for (String line : lines) {

                Integer[] cellID = new Integer[6];
                String[] data = line.split(delimiter);

                a++;

                if (a == 1) {
                    continue;
                }

                for (int i = 0; i < cellID.length - 1; i++) {
                    cellID[i] = Integer.parseInt(data[i + 5]);
                }

                if (lines.length == 2) {
                    cellID[4] = 1;
                }

                outlinesImage.add((cellID));   // X, Y, Width, Height, Category, Match
            }

            outlinesAll.add(outlinesImage);  // fill all outlines

        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    private void readLabeled(String fileNameJSON) {

        LinkedList<Integer[]> labeledImage = new LinkedList<>();
        var pathLabeled = pathLabels + "\\" + fileNameJSON + ".json";

        try (FileReader reader = new FileReader(pathLabeled)) {

            JSONObject jsonObject = new JSONObject(new JSONTokener(reader));
            int cellNumbers = jsonObject.getInt("Cell Numbers");

            for (int cellId = 0; cellId < cellNumbers; cellId++) {

                JSONObject cellJson = jsonObject.getJSONObject("Cell_" + cellId);
                int xmin = cellJson.getInt("x1");
                int xmax = cellJson.getInt("x2");
                int ymin = cellJson.getInt("y1");
                int ymax = cellJson.getInt("y2");
                String wbcType = cellJson.getString("Label2");

                Integer[] labeledCellID = new Integer[]{xmin, ymin, xmax - xmin, ymax - ymin, cellId + 1, null}; // X,Y,Width,Height
                labeledImage.add(labeledCellID);
            }

            labeledAll.add(labeledImage);

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void joinOutlines(LinkedList<Integer[]> outlineImg) {

        if (outlineImg.size() > 1) {

            int thresholdDistance =
                    Integer.parseInt(Common.argumentsMap.get(thresholdDistanceCellPxParamName) != null ?
                    Common.argumentsMap.get(thresholdDistanceCellPxParamName) : thresholdDistanceCellPxParamValue);

            int lastSetClass = 1;
            outlineImg.get(0)[4] = lastSetClass;

            for (int i = 0; i < outlineImg.size() - 1; i++) {

                lastSetClass = outlineImg.get(i)[4];

                for (int j = i + 1; j < outlineImg.size(); j++) {
                    Rectangle rect1 = new Rectangle((outlineImg.get(i))[0], (outlineImg.get(i))[1],
                            (outlineImg.get(i))[2], (outlineImg.get(i)[3]));
                    Rectangle rect2 = new Rectangle((outlineImg.get(j))[0], (outlineImg.get(j))[1],
                            (outlineImg.get(j))[2], (outlineImg.get(j)[3]));

                    // check if the two rectangles are close to each other
                    Point center1 = new Point(rect1.x + rect1.width / 2, rect1.y + rect1.height / 2);
                    Point center2 = new Point(rect2.x + rect2.width / 2, rect2.y + rect2.height / 2);

                    double distance = center1.distance(center2);

                    if (distance < thresholdDistance) {
                        // the two rectangles are close to each other
                        outlineImg.get(j)[4] = outlineImg.get(i)[4];
                    } else {
                        lastSetClass++;
                        outlineImg.get(j)[4] = lastSetClass;
                    }
                }
            }
        }
    }

    private void findMatch() {

        for (int i = 0; i < outlinesAll.size(); i++) {

            int order = 1;
            LinkedList<Integer[]> labelImage = labeledAll.get(i);
            LinkedList<Integer[]> outlineImage = outlinesAll.get(i);

            if (outlineImage.size() > 0 && labelImage.size() > 0 ) {

                if (outlineImage.getLast()[4] >= labelImage.size()) {
                    order = 0;
                    compareCells(order, outlineImage, labelImage);
                } else
                    compareCells(order, labelImage, outlineImage);

            } else {

                int countOutlines = (outlineImage.size() == 0) ? 0 : outlineImage.getLast()[4];
                int countLabels = labelImage.size();
                Integer[] statsImage = new Integer[]{0, countOutlines, countLabels};
                matches.add(statsImage);
            }
        }

        // make stats
        calculateStats();
    }

    private void calculateStats() {
        ArrayList<Double> percentOkRatio = new ArrayList<>();
        ArrayList<Double> percentOkMoreRatio = new ArrayList<>();
        ArrayList<Double> matchMore = new ArrayList<>();
        ArrayList<Double> matchOK = new ArrayList<>();
        ArrayList<Double> matchLess = new ArrayList<>();
        ArrayList<ArrayList<Double>> datasetToCsv = new ArrayList<>();
        ArrayList<String> classes = new ArrayList<>();
        int findOKImg = 0;
        int findOK_More_Img = 0;
        int findLessImg = 0;
        int findMoreImg = 0;

        for (Integer[] match : matches) {
            if (match[0] >= 0 && match[2]==0) findOK_More_Img++;
            if (match[0] >= 0 && match[1]==0 && match[2]==0) findOKImg++;
            if (match[1] > 0) findMoreImg++;
            if (match[2] > 0) findLessImg++;
            percentOkMoreRatio.add((match[0]*100.0)/(match[0] + match[2]));
            percentOkRatio.add((match[0]*100.0)/(match[0] + match[2]));
            matchOK.add((double)match[0]);
            matchMore.add((double)match[1]);
            matchLess.add((double)match[2]);
        }
        double okMoreRatio = (findOK_More_Img*100.0) / fileNameList.size();
        double okRatio = (findOKImg*100.0) / fileNameList.size();
        double lessRatio = (findLessImg*100.0) / fileNameList.size();
        double moreRatio = (findMoreImg*100.0) / fileNameList.size();

        System.out.println(datasetName + ":");
        System.out.printf("Identifikované (Správne+Neoznačené): %.2f\n",okMoreRatio);
        System.out.printf("Identifikované (Správne): %.2f\n",okRatio);
        System.out.printf("Neidentifikované (Označené): %.2f\n",lessRatio);
        System.out.printf("Neoznačené (Identifikované): %.2f\n",moreRatio);

        plotBarChart(percentOkRatio,"Percentuálne vyjadrenie pomeru Identifikovaných (Správne) ku všetkým označeným leukocytom: " + datasetName, datasetName);
        plotStackedBarChart(matchOK, matchMore, matchLess, "Počet leukocytov v kategóriách", datasetName);

        datasetToCsv.add(percentOkRatio);
        classes.add("Identifikované");
        exportToCsv(datasetToCsv, classes, datasetName, "identified_"+datasetName+".csv", ";");
        datasetToCsv = new ArrayList<>();
        datasetToCsv.add(matchOK);
        datasetToCsv.add(matchMore);
        datasetToCsv.add(matchLess);

        classes = new ArrayList<>();
        classes.add("Identifikované (Správne)");
        classes.add("Neidentifikované (Označené)");
        classes.add("Neoznačené (Identifikované)");

        exportToCsv(datasetToCsv, classes , datasetName, "categories_count_"+datasetName+".csv", ";");
    }

    private void exportToCsv(ArrayList<ArrayList<Double>>data, ArrayList<String> rowNames,  String datasetName,
                             String fileName, String delimiter) {

        Locale.setDefault(Locale.GERMANY);

        try {

            PrintStream writer = new PrintStream(datasetName + '_' + fileName, "UTF-8");
            String csvContent = "";
            String header = "";
            csvContent += "" + delimiter;

            for (int i = 1; i < data.get(0).size()+1 ; i++) {

                if (i< data.get(0).size())
                    header = header +  fileNameList.get(i-1)+".jpg" + delimiter;
                else
                    header = header + fileNameList.get(i-1)+".jpg";
            }
            csvContent += header + "\n";
            int count = 0;

            for (ArrayList<Double> dataset : data) {

                csvContent += rowNames.get(count) + delimiter;
                count++;

                for (Double d : dataset) {
                    csvContent += String.format("%.2f", d) + delimiter;
                }

                csvContent += "\n";
            }

            writer.print(csvContent);
            writer.flush();
            writer.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void plotStackedBarChart(ArrayList<Double> matchOK, ArrayList<Double> matchMore, ArrayList<Double> matchLess,
                                     String chartTitle, String datasetNameParam) {

        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        for(int i = 0; i < matchOK.size(); i++) {

            dataset.addValue(matchOK.get(i), "Identifikované (Správne)", this.fileNameList.get(i) + ".jpg");
            dataset.addValue(matchMore.get(i), "Neoznačené (Identifikované)", this.fileNameList.get(i) + ".jpg");
            dataset.addValue(matchLess.get(i), "Neidentifikované (Označené)", this.fileNameList.get(i) + ".jpg");
        }

        // create chart
        JFreeChart chart = ChartFactory.createStackedBarChart(
                chartTitle +": "+ datasetNameParam, // chart title
                "Snímky", // domain axis label
                "Počet", // range axis label
                dataset, // data
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        // display chart
        ChartFrame frame = new ChartFrame("Skladaný stĺpcový graf", chart);
        frame.setVisible(true);
        frame.setSize(500, 500);

        try {
            ChartUtilities.saveChartAsPNG(new File("categories_count_" + datasetName + ".png"), chart, 1000, 600);
        } catch (IOException e) {
            System.out.println("Error saving chart to image.");
        }

    }

    private void plotBarChart(ArrayList<Double> data, String chartTitle, String dataName) {

        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        for(int i = 0; i < data.size(); i++) {
            dataset.addValue(data.get(i), "Identifikované (Správne)", this.fileNameList.get(i) + ".jpg");
        }

        // create chart
        JFreeChart chart = ChartFactory.createBarChart(
                chartTitle, // chart title
                "Snímky", // x-axis label
                "Úspešnosť [%]", // y-axis label
                dataset, // data
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        // display chart
        ChartFrame frame = new ChartFrame("Graf", chart);
        frame.pack();
        frame.setVisible(true);

        try {
            ChartUtilities.saveChartAsPNG(new File("identified_" + datasetName + ".png"), chart, 1000, 600);
        } catch (IOException e) {
            System.out.println("Error saving chart to image.");
        }
    }

    private void compareCells(int order, LinkedList<Integer[]> large, LinkedList<Integer[]> small) {

        // 0 - large: outline , 1 - large: labeled
        int lastCellCat = 0;
        Integer[] statsImage = new Integer[]{0, 0, 0};   // [[find ok] [more] = not in labeled [less] = not in outlines]

        for (Integer[] cellsLarge : large) {

                if (cellsLarge[4] > lastCellCat) {
                    lastCellCat = cellsLarge[4];

                    int lastCellCatSmall = 0;

                    for (Integer[] cellsSmall : small) {

                        boolean matchStatus = false;

                        if (cellsSmall[4] == lastCellCatSmall) {
                            continue;
                        }

                        lastCellCatSmall = cellsSmall[4];

                        if (cellsSmall[5] != null) {
                            continue;
                        }

                        if (order == 0) {
                            matchStatus = isInside(cellsSmall[0], cellsSmall[1], cellsSmall[2], cellsSmall[3],
                                    cellsLarge[0], cellsLarge[1], cellsLarge[2], cellsLarge[3]);
                        } else {
                            matchStatus = isInside(cellsLarge[0], cellsLarge[1], cellsLarge[2], cellsLarge[3],
                                    cellsSmall[0], cellsSmall[1], cellsSmall[2], cellsSmall[3]);
                        }

                        if (matchStatus) {
                            cellsSmall[5] = 1;
                            cellsLarge[5] = 1;
                            statsImage[0]++;  // mark match
                            break;
                        }
                    }
                } else
                    continue;  // drop if same category
        }

        int countLarge = 0;

        for (Integer[] cellLarge : large)
            if (cellLarge[5] == null)
                countLarge++;

        int countSmall = 0;

        for (Integer[] cellSmall : small)
            if (cellSmall[5] == null)
                countSmall++;

        // 0 - large: outline , 1 - large: labeled
        statsImage[1 + order] = countLarge;
        statsImage[2 - order] = countSmall;
        matches.add(statsImage);
    }

        public static boolean isInside (int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2){
            // calculate the edges of the first rectangle
            int left1 = x1;
            int right1 = x1 + w1;
            int top1 = y1;
            int bottom1 = y1 + h1;

            // calculate the edges of the second rectangle
            int left2 = x2;
            int right2 = x2 + w2;
            int top2 = y2;
            int bottom2 = y2 + h2;

            // check if the second rectangle is inside the first one
            if (left2 >= left1 && right2 <= right1 && top2 >= top1 && bottom2 <= bottom1) {
                return true;
            }
            return false;
        }

        public boolean run (String datasetNameParam){

            pathOutlines = Common.argumentsMap.get(statsOutputDirParamName);
            pathLabels = Common.argumentsMap.get(Common.annotationsPathToParamName);
            outlinesAll = new ArrayList<>();
            labeledAll = new ArrayList<>();
            matches = new ArrayList<>();
            datasetName = datasetNameParam;

            String pathImagesLabeled = Common.argumentsMap.get(Common.sampleDirParamName);

            File sampleDirectory = new File(pathImagesLabeled);
            this.fileNameList = Common.getFileNameList(sampleDirectory, false);

            for (String fileName : fileNameList) readOutlines(fileName); // read all CSV outlines
            for (String fileName : fileNameList) readLabeled(fileName); // read all JSON
            for (LinkedList<Integer[]> outlineImg : outlinesAll) joinOutlines(outlineImg); // join outlines

            // check matches between labeled and category
            findMatch();

            return true;
        }

    }

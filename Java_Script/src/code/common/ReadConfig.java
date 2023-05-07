package code.common;

import java.io.*;

/**
 * Class ReadConfig
 * reads the configuration file
 */
public class ReadConfig {

    /**
     * readConfigFile - reads the configuration file stored in resources/config/
     */
    public void readConfigFile() {

        // read the config file
        File configFile = new File(System.getProperty("user.dir") +"\\resources\\config\\ImageJAutoRunConfigFile.txt");
        StringBuilder content = new StringBuilder();

        try {

            BufferedReader reader = new BufferedReader(new FileReader(configFile));
            String line;

            // read config file
            while ((line = reader.readLine()) != null) {
                content.append(line);
                content.append(System.lineSeparator());
            }
            reader.close();

        } catch (FileNotFoundException e) {
            System.out.println("Config file not found!" + e.getMessage());
            return;
        } catch (IOException e) { // file reading exception
            System.out.println("Could not read config file!" + e.getMessage());
            return;
        }

        if(!content.toString().contains(Common.sampleDirParamName) ||
                !content.toString().contains(Common.appDirParamName) ||
                !content.toString().contains(Common.outputDirParamName)){
            System.out.println("Config file must contain sampleDirParam, outputDirParam and appDirParam variables with their values!");
            return;
        }

        // read arguments
        String[] configVariables = content.toString().split(System.lineSeparator());

        // read arguments
        for (String configVariable : configVariables) {

            String[] variable = configVariable.split("=");

            if (variable.length != 2) {
                System.out.println("Variables must contain their values!");
                return;
            }

            Common.argumentsMap.put(variable[0], variable[1]);
        }
    }
}

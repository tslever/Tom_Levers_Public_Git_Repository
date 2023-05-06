/*
 * SubmarineDiagnosticReportInterpreter
 * author Tom Lever
 * version 0.0
 * since 01/26/22
 */

package com.tsl.interview.cnri;

import com.tsl.interview.cnri.validator.BinaryNumbersValidator;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class SubmarineDiagnosticReportInterpreter {
    
    public static void main(String[] args) throws Exception {
        String pathToDiagnosticReport = args[0];
        List<String> binaryNumbers = Files.readAllLines(Paths.get(pathToDiagnosticReport));
        BinaryNumbersValidator.validate(binaryNumbers);
        SubmarineStatisticsGenerator submarineStatisticsGenerator = new SubmarineStatisticsGenerator(binaryNumbers);
        System.out.printf("The power consumption of the submarine corresponding to the diagnostic report at\n\"%s\"\nis %d.\n\n",
            pathToDiagnosticReport,
            submarineStatisticsGenerator.getPowerConsumption());
        System.out.println("The life-support rating of the submarine corresponding to the diagnostic report at\n\""
            + pathToDiagnosticReport + "\"\nis " + submarineStatisticsGenerator.getLifeSupportRating() + ".");
        
    }
}
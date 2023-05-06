/*
 * BinaryNumbersValidatorTest
 * author Tom Lever
 * version 0.0
 * since 01/26/22
 */

package com.tsl.utilitiesforsubmarinediagnosticreportinterpreter;

import com.tsl.interview.cnri.validator.ZeroBinaryNumbersException;
import com.tsl.interview.cnri.validator.BinaryNumbersValidator;
import com.tsl.interview.cnri.validator.InconsistentNumberOfBitsException;
import com.tsl.interview.cnri.validator.InvalidCharacterException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class BinaryNumbersValidatorTest {
        
    @Test
    void testValidatingWithInconsistentNumberOfBits() throws Exception {
        System.out.println("Running testValidatingWithInconsistentNumberOfBits");
        String pathToDiagnosticReport = "resources/The_Diagnostic_Report_With_An_Inconsistent_Number_Of_Bits.txt";
        try {
            BinaryNumbersValidator.validate(Files.readAllLines(Paths.get(pathToDiagnosticReport)));
            fail();
        } catch (InconsistentNumberOfBitsException inconsistentNumberOfBitsException) {
            System.out.println("The extractor found an inconsistent number of bits when trying to read a line.\n");
        }
    }
    
    @Test
    void testValidatingWithInvalidCharacter() throws Exception {
        System.out.println("Running testValidatingWithAnInvalidCharacter");
        String pathToDiagnosticReport = "resources/The_Diagnostic_Report_With_An_Invalid_Character.txt";
        try {
            BinaryNumbersValidator.validate(Files.readAllLines(Paths.get(pathToDiagnosticReport)));
            fail();
        } catch (InvalidCharacterException invalidCharacterException) {
            System.out.println("The extractor found an invalid character when trying to read a line.\n");
        }
    }

    @Test
    void testValidatingWithZeroBits() throws Exception {
        System.out.println("Running testValidatingWithZeroBits");
        String pathToDiagnosticReport = "resources/The_Diagnostic_Report_With_Zero_Bits.txt";
        try {
            BinaryNumbersValidator.validate(Files.readAllLines(Paths.get(pathToDiagnosticReport)));
            fail();
        } catch (ZeroBinaryNumbersException zeroBinaryNumbersException) {
            System.out.println("The extractor found zero binary numbers.\n");
        }
    }
    
    @Test
    void testWellBehavedExtracting() throws Exception {
        System.out.println("Running testWellBehavedExtracting");
        String pathToDiagnosticReport = "resources/The_Diagnostic_Report_With_Few_Bits.txt";
        BinaryNumbersValidator.validate(Files.readAllLines(Paths.get(pathToDiagnosticReport)));
        System.out.println("Extracting a list of binary numbers succeeded.\n");
    }
}
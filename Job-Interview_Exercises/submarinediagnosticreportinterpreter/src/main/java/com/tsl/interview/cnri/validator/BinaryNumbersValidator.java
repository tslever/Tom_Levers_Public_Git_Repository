/*
 * ExtractorOfBinaryNumbers
 * author Tom Lever
 * version 0.0
 * since 01/26/22
 */

package com.tsl.interview.cnri.validator;

import java.util.List;

public class BinaryNumbersValidator {
    private static void checkCharacters(String binaryNumber) throws InvalidCharacterException {
        for (int i = 0; i < binaryNumber.length(); i++) {
            char ch = binaryNumber.charAt(i);
            if ((ch != '0') && (ch != '1')) {
                throw new InvalidCharacterException("A proposed binary number " + binaryNumber + " contains an invalid character " + ch);
            }
        }
    }
    
    private static void checkNumberOfBits(String binaryNumber, int expectedNumberOfBits) throws InconsistentNumberOfBitsException {
        if (binaryNumber.length() != expectedNumberOfBits) {
            throw new InconsistentNumberOfBitsException(
                "A binary number differs in its number of bits from the first read binary number.");
        }
    }
    
    public static void validate(List<String> binaryNumbers) throws
            ZeroBinaryNumbersException, InconsistentNumberOfBitsException, InvalidCharacterException {
        if (binaryNumbers.isEmpty()) {
            throw new ZeroBinaryNumbersException(
                    "BinaryNumbersValidator found zero binary numbers.");
        }
        int expectedNumberOfBits = binaryNumbers.get(0).length();
        for (String binaryNumber : binaryNumbers) {
            checkCharacters(binaryNumber);
            checkNumberOfBits(binaryNumber, expectedNumberOfBits);
        }
    }
}
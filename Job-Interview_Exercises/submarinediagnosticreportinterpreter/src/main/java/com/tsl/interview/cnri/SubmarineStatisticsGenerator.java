/*
 * GeneratorOfSubmarineStatistics
 * author Tom Lever
 * version 0.0
 * since 01/26/22
 */

package com.tsl.interview.cnri;

import java.util.ArrayList;
import java.util.List;

class SubmarineStatisticsGenerator {
    private final List<String> binaryNumbers;
    private final int bitCount;
    
    public SubmarineStatisticsGenerator(List<String> binaryNumbers) {
        this.binaryNumbers = binaryNumbers;
        this.bitCount = binaryNumbers.get(0).length();
    }
    
    private BitFrequencies getBitFrequenciesAtIndex(List<String> binaryNumbers, int indexOfBit) {
        BitFrequencies result = new BitFrequencies();
        for (String binaryNumber : binaryNumbers) {
            if (binaryNumber.charAt(indexOfBit) == '0') {
                result.zeros++;
            } else {
                result.ones++;
            }
        }
        return result;
    }
    
    private int getGasRating(Gas gas) {
        List<String> remainingBinaryNumbers = this.binaryNumbers;
        for (int i = 0; i < this.bitCount; i++) {
            BitFrequencies frequencies = this.getBitFrequenciesAtIndex(remainingBinaryNumbers, i);
            char significantBit;
            if (gas == Gas.CARBON_DIOXIDE) {
                significantBit = (frequencies.ones < frequencies.zeros) ? '1' : '0';
            } else if (gas == Gas.OXYGEN) {
                significantBit = (frequencies.ones >= frequencies.zeros) ? '1' : '0';
            } else {
                throw new AssertionError("Can't happen");
            }
//            int theIndex = i;
//            List<String> winnowedBinaryNumbers = remainingBinaryNumbers.stream()
//                    .filter(binaryNumber -> binaryNumber.charAt(theIndex) == significantBit)
//                    .collect(Collectors.toList());
            List<String> winnowedBinaryNumbers = winnowToNumbersMatchingBitAtIndex(remainingBinaryNumbers, significantBit, i);
            if (winnowedBinaryNumbers.isEmpty()) {
                return Integer.parseInt(remainingBinaryNumbers.get(0), 2);
            } else {
                remainingBinaryNumbers = winnowedBinaryNumbers;
            }
        }
        return Integer.parseInt(remainingBinaryNumbers.get(0), 2);
    }

    private List<String> winnowToNumbersMatchingBitAtIndex(List<String> binaryNumbers, char bit, int index) {
        List<String> winnowedBinaryNumbers = new ArrayList<>();
        for (String binaryNumber : binaryNumbers) {
            if (binaryNumber.charAt(index) == bit) {
                winnowedBinaryNumbers.add(binaryNumber);
            }
        }
        return winnowedBinaryNumbers;
    }

    private int getEpsilonRate() {
        String epsilonRate = "";
        for (int i = 0; i < this.bitCount; i++) {
            BitFrequencies frequencies = this.getBitFrequenciesAtIndex(this.binaryNumbers, i);
            epsilonRate += (frequencies.ones < frequencies.zeros) ? "1" : "0";
        }
        return Integer.parseInt(epsilonRate, 2);
    }
    
    private int getGammaRate() {
        String gammaRate = "";
        for (int i = 0; i < this.bitCount; i++) {
            BitFrequencies frequencies = this.getBitFrequenciesAtIndex(this.binaryNumbers, i);
            gammaRate += (frequencies.ones >= frequencies.zeros) ? "1" : "0";
        }
        return Integer.parseInt(gammaRate, 2);
    }

    private int getOxygenGeneratorRating() {
        return getGasRating(Gas.OXYGEN);
    }

    private int getCarbonDioxideScrubberRating() {
        return getGasRating(Gas.CARBON_DIOXIDE);
    }

    public int getPowerConsumption() {
        return getEpsilonRate() * getGammaRate();
    }

    public int getLifeSupportRating() {
        return getCarbonDioxideScrubberRating() * getOxygenGeneratorRating();
    }

    private static class BitFrequencies {
        public int zeros = 0;
        public int ones = 0;
    }
}
package Com.TSL.GcfAlgorithmEvaluationUtilities;


import java.lang.Math;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.LogRecord;


/**
 * Class GCDFinder represents an object that finds the Greatest Common Denominator of two integers.
 * @version 0.0
 * @author Tom Lever
 */
public class TomLeversGCDFinder {

    // Logger logger allows logging output to console.
    private static Logger logger = Logger.getLogger(TomLeversGCDFinder.class.getName());

    /**
     * Method main represents the entrypoint of GCDFinder's functionality.
     * @param String[] args
     * @throws InputsNotEqualToTwoException
     * @throws NumberFormatException
     */
    public static void main(String[] args) throws InputsNotEqualToTwoException, NumberFormatException {

        InputParsingUtilities.checkNumberOfInputsIn(args);
        int[] arrayOfIntegers = InputParsingUtilities.convertToArrayOfIntegers(args);
        InputParsingUtilities.logIntegersIn(arrayOfIntegers);
        int gcd = GCDCalculatingUtilities.calculateGCDBasedOn(arrayOfIntegers);
        GCDCalculatingUtilities.log(gcd);
    }

    /**
     * Class InputsNotEqualToTwoException is the template for exceptions thrown when entrypoint main does not receive two inputs.
     * @version 0.0
     * @author Tom Lever
     */
    private static class InputsNotEqualToTwoException extends Exception {
        public InputsNotEqualToTwoException(String errorMessage) {
            super(errorMessage);
        }
    }

    /**
     * Class InputParsingUtilities encapsulates methods used to parse a group of input arguments.
     * @version 0.0
     * @author Tom Lever
     */
    private static class InputParsingUtilities {

        /**
         * Method checkNumberOfInputsIn throws an InputsNotEqualToTwoExceptions
         * if the number of input arguments in an input array is not equal to two.
         * @param String[] args
         * @throws InputsNotEqualToTwoException
         */
        public static void checkNumberOfInputsIn(String[] args) throws InputsNotEqualToTwoException {
            if (args.length != 2) throw new InputsNotEqualToTwoException("Length of argument string is not equal to 2.");
        }

        /**
         * Method convertToArrayOfIntegers converts an array of input arguments to an array of integers.
         * @param String[] args
         * @return int[] arrayOfIntegers
         * @throws NumberFormatException
         */
        public static int[] convertToArrayOfIntegers(String[] args) throws NumberFormatException {

            int[] arrayOfIntegers = new int[args.length];
            for (int i = 0; i < args.length; i++) {
                arrayOfIntegers[i] = Integer.parseInt(args[i]);
            }

            return arrayOfIntegers;
        }
        
        /**
         * Method logIntegersIn logs the integers in an array of integers to console.
         * @param int[] arrayOfIntegers
         */
        public static void logIntegersIn(int[] arrayOfIntegers) {

            String logOfIntegers = "Integers: [";
            for (int i = 0; i < arrayOfIntegers.length-1; i++) {
                logOfIntegers += arrayOfIntegers[i] + ", ";
            }
            logOfIntegers += arrayOfIntegers[arrayOfIntegers.length-1] + "]";

            logger.log(new LogRecord(Level.INFO, logOfIntegers));
        }
    }

    /**
     * GCDCalculatingUtilities encapsulates methods used to calculate the GCD of integers in a group.
     * @version 0.0
     * @author Tom Lever
     */
    private static class GCDCalculatingUtilities {

        /**
         * Method applyModernEuclideanAlgorithmTo applies the "Modern Euclidean Algorithm" from the following source
         * to an array of magnitudes of integers.
         * Knuth, Donald E. (1998). "4.5.2. The Greatest Common Divisor". The Art of Computer Programming, Volume 2,
         * Seminumerical Algorithms, Third Edition.
         * @param int[] arrayOfMagnitudes
         * @return int[] arrayOfMagnitudes
         */
        public static int[] applyModernEuclideanAlgorithmTo(int[] arrayOfMagnitudes) {

            int remainder;
            while (!isIntegerZeroIn(arrayOfMagnitudes)) {
                remainder = arrayOfMagnitudes[0] % arrayOfMagnitudes[1];
                arrayOfMagnitudes[0] = arrayOfMagnitudes[1];
                arrayOfMagnitudes[1] = remainder;
            }

            return arrayOfMagnitudes;
        }

        /**
         * Method calculateGCDBasedOn calculates a Greatest Common Divisor based on an array of integers.
         * @param int[] arrayOfIntegers
         * @return int <greatest magnitude in array of magnitudes>
         */
        public static int calculateGCDBasedOn(int[] arrayOfIntegers) {

            int[] arrayOfMagnitudes = convertToArrayOfMagnitudes(arrayOfIntegers);
            int[] array = applyModernEuclideanAlgorithmTo(arrayOfMagnitudes);
            return (array[0] > array[1]) ? array[0] : array[1];
        }

        /**
         * Method convertToArrayOfMagnitudes converts an array of integers into an array of magnitudes.
         * @param int[] arrayOfIntegers
         * @return int[] arrayOfMagnitudes
         */
        public static int[] convertToArrayOfMagnitudes(int[] arrayOfIntegers) {

            int[] arrayOfMagnitudes = new int[arrayOfIntegers.length];
            for (int i = 0; i < arrayOfIntegers.length; i++) {
                arrayOfMagnitudes[i] = Math.abs(arrayOfIntegers[i]);
            }

            return arrayOfMagnitudes;
        }

        /**
         * Method isIntegerZeroIn evaluates whether there is an integer in an array of integers that is zero.
         * @param int[] arrayOfIntegers
         * @return boolean <indication of whether an integer is zero>
         */
        public static boolean isIntegerZeroIn(int[] arrayOfIntegers) {
            return ((arrayOfIntegers[0] == 0) || (arrayOfIntegers[1] == 0));
        }

        /**
         * Method log logs a Greatest Common Divisor to console.
         * @param int gcd
         */
        public static void log(int gcd) {
            logger.log(new LogRecord(Level.INFO, "Greatest Common Divisor: " + gcd));
        }
    }
}
using System;
using System.Collections.Generic;
using System.Text;


namespace UtilitiesForADecimalToBinaryConverter
{
    /// <summary>
    /// TheDecimalToBinaryConverter encapsulates the entry point of this program,
    /// which converts a hard-coded decimal integer to its binary form.
    /// 
    /// Author: Tom Lever
    /// Version: 0.0
    /// Since: 08/17/21
    /// </summary>

    class TheDecimalToBinaryConverter
    {
        /// <summary>
        /// Main represents the entry point of this program, which converts a hard-coded
        /// decimal integer to its binary form.
        /// </summary>
        /// <param name="args"></param>

        static void Main(string[] args)
        {

            int theIntegerToConvertFromDecimalToBinary = -8;

            Stack<int> theStackOfBinaryDigits = theStackOfBinaryDigitsCorrespondingTo(theIntegerToConvertFromDecimalToBinary);

            Console.WriteLine(
                "The binary representation of the decimal integer " +
                theIntegerToConvertFromDecimalToBinary +
                ", including a leading sign bit, is " +
                theBinaryRepresentationOf(theStackOfBinaryDigits) + "."
            );

            Console.Read();
        }


        /// <summary>
        /// theBinaryRepresentationOf provides the binary representation of an stack of
        /// binary digits.
        /// </summary>
        /// <param name="theStackOfBinaryDigits"></param>
        /// <returns></returns>

        private static string theBinaryRepresentationOf(Stack<int> theStackOfBinaryDigits)
        {
            StringBuilder theStringBuilder = new StringBuilder();

            while (theStackOfBinaryDigits.Count > 0)
            {
                theStringBuilder.Append(theStackOfBinaryDigits.Pop());
            }

            return theStringBuilder.ToString();
        }


        /// <summary>
        /// theSignOf provides the sign of an integer.
        /// </summary>
        /// <param name="theInteger"></param>
        /// <returns></returns>

        private static int theSignOf(int theInteger)
        {
            int theSignOfTheInteger = Math.Sign(theInteger);

            if (theSignOfTheInteger == 0)
            {
                theSignOfTheInteger = 1;
            }
            else if (theSignOfTheInteger == -1)
            {
                theSignOfTheInteger = 0;
            }

            return theSignOfTheInteger;
        }


        /// <summary>
        /// theStackOfBinaryDigitsCorrespondingTo provides the stack of binary digits
        /// corresponding to an integer.
        /// </summary>
        /// <param name="theInteger"></param>
        /// <returns></returns>

        private static Stack<int> theStackOfBinaryDigitsCorrespondingTo(int theInteger)
        {
            int theSignOfTheIntegerToConvert = theSignOf(theInteger);

            Stack<int> theStackOfBinaryDigits = new Stack<int>();

            int theDividend = Math.Abs(theInteger);
            int theRemainder = -1;

            while (theDividend > 0)
            {
                theRemainder = theDividend % 2;
                theStackOfBinaryDigits.Push(theRemainder);

                theDividend /= 2;
            }

            theStackOfBinaryDigits.Push(theSignOfTheIntegerToConvert);

            return theStackOfBinaryDigits;
        }

    }

}

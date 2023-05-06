using System.Text;


namespace UtilitiesForAnEvaluatorOfWhetherOrNotAStringContainsAnotherString
{

    /// <summary>
    /// TheEvaluatorOfWhetherOrNotAStringContainsAnotherString encapsulates the entry point
    /// of this program, which determines whether a string contains another string.
    /// </summary>

    class TheEvaluatorOfWhetherOrNotAStringContainsAnotherString
    {

        /// <summary>
        /// Main is the entry point of this program, which determines whether a string
        /// contains another string.
        /// </summary>
        /// <param name="args"></param>

        static void Main(string[] args)
        {
            string string1 = "cat";
            string string2 = "thisisfox";
            System.Console.Write("Considering String 1, \"" + string1 + "\", and String 2, \"" + string2 + "\", ");

            if (string1.Length > string2.Length)
            {
                System.Console.WriteLine("String 1 is not contained in String 2.");
                return;
            }

            bool stringTwoContainsStringOne = false;

            for (int i = 0; i <= string2.Length - string1.Length; i++)
            {
                if (areEqual(string1, theSubstringOfAStringStartingAtAnIndexWithALengthGiven(string2, i, string1.Length)))
                {
                    stringTwoContainsStringOne = true;
                    break;
                }
            }

            System.Console.WriteLine("\"" + string1 + "\" is " + ((stringTwoContainsStringOne) ? "" : "not ") + "in \"" + string2 + "\".");
            System.Console.Read();

        }


        /// <summary>
        /// areEqual indicates whether or not two strings are equal.
        /// </summary>
        /// <param name="theFirstString"></param>
        /// <param name="theSecondString"></param>
        /// <returns></returns>

        private static bool areEqual(string theFirstString, string theSecondString)
        {
            if (theFirstString.Length != theSecondString.Length)
            {
                return false;
            }

            for (int i = 0; i < theFirstString.Length; i++)
            {
                if (!(theFirstString[i].Equals(theSecondString[i])))
                {
                    return false;
                }
            }

            return true;
        }


        /// <summary>
        /// theSubstringOfAStringStartingAtAnIndexWithALengthGiven provides a substring of
        /// a string starting at an index with a length, given the substring, the index, and
        /// the length.
        /// </summary>
        /// <param name="theSecondString"></param>
        /// <param name="theStartingIndex"></param>
        /// <param name="theLengthOfTheFirstString"></param>
        /// <returns></returns>

        private static string theSubstringOfAStringStartingAtAnIndexWithALengthGiven(string theString, int theIndex, int theLength)
        {
            StringBuilder theStringBuilderForTheSubstring = new StringBuilder();

            for (int i = theIndex; i < theIndex + theLength; i++)
            {
                theStringBuilderForTheSubstring.Append(theString[i]);
            }

            return theStringBuilderForTheSubstring.ToString();
        }

    }

}

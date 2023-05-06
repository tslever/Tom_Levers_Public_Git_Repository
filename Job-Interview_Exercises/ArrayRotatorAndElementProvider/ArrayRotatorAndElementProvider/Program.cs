using System.Collections.Generic;


namespace UtilitiesForTheArrayRotatorAndElementProvider
{

    /// <summary>
    /// TheTesterForAnArrayRotatorAndElementProvider encapsulates the entry point of this
    /// program, which sets up and tests methods of an array rotator and element provider.
    /// </summary>

    class TheTesterForAnArrayRotatorAndElementProvider
    {

        /// <summary>
        /// Main is the entry point of this program, which sets up and tests methods of an
        /// array rotator and element provider.
        /// </summary>
        /// <param name="args"></param>

        static void Main(string[] args)
        {

            AnArrayRotatorAndElementProvider<int> theArrayRotatorAndElementProvider =
                new AnArrayRotatorAndElementProvider<int>(3);

            for (int i = 0; i < theArrayRotatorAndElementProvider.providesItsNumberOfElements(); i++)
            {
                theArrayRotatorAndElementProvider.setsItsElementGiven(i, i + 1);
            }


            theArrayRotatorAndElementProvider.Rotate(0);

            theArrayRotatorAndElementProvider.displaysItsElements();

            theArrayRotatorAndElementProvider.Rotate(1);

            theArrayRotatorAndElementProvider.displaysItsElements();

            theArrayRotatorAndElementProvider.Rotate(1);

            theArrayRotatorAndElementProvider.displaysItsElements();

            theArrayRotatorAndElementProvider.Rotate(1);

            theArrayRotatorAndElementProvider.displaysItsElements();


            theArrayRotatorAndElementProvider.Rotate(2);

            theArrayRotatorAndElementProvider.displaysItsElements();


            System.Console.Read();

        }

    }


    /// <summary>
    /// AnArrayRotatorAndElementProvider represents the structure for an array rotator and
    /// element provider, which may set its elements, provide its number of elements,
    /// provide an element, display its elements, and rotate its elements.
    /// </summary>
    /// <typeparam name="T"></typeparam>

    class AnArrayRotatorAndElementProvider<T>
    {
        T[] array;
        int indexOfTheFirstElement;


        /// <summary>
        /// AnArrayRotatorAndElementProvider(int theNumberOfElementsToUse) is the
        /// one-parameter constructor for AnArrayRotatorAndElementProvider, which sets the
        /// reference of this array rotator and element provider to an array of elements of
        /// type T to a new array of elements of type T with a capacity of a provided number
        /// of elements.
        /// Precondition: theNumberOfElementsToUse must not be negative.
        /// </summary>
        /// <param name="theNumberOfElementsToUse"></param>

        public AnArrayRotatorAndElementProvider(int theNumberOfElementsToUse)
        {
            this.array = new T[theNumberOfElementsToUse];
        }


        /// <summary>
        /// displaysItsElements displays the elements of this array rotator and provider.
        /// Precondition: The ToString method of type T must result in a pretty
        /// representation of an object of type T.
        /// </summary>

        public void displaysItsElements()
        {
            for (int i = 0; i < this.providesItsNumberOfElements() - 1; i++)
            {
                System.Console.Write(this.GetAt(i) + ", ");
            }
            System.Console.WriteLine(this.GetAt(this.providesItsNumberOfElements() - 1));
        }


        /// <summary>
        /// GetAt provides the element of this array rotator and element provider at the
        /// provided index.
        /// Precondition: theIndexOfTheElementToGet must not be negative.
        /// </summary>
        /// <param name="theIndexOfTheElementToGet"></param>
        /// <returns></returns>

        public T GetAt(int theIndexOfTheElementToGet)
        {
            return this.array[(this.indexOfTheFirstElement + theIndexOfTheElementToGet) % this.array.Length];
        }


        /// <summary>
        /// providesItsNumberOfElements provides the number of elements for this array
        /// rotator and provider.
        /// </summary>
        /// <returns></returns>

        public int providesItsNumberOfElements()
        {
            return this.array.Length;
        }


        /// <summary>
        /// Rotate cycles the index of the first element of this array rotator and provider
        /// by a provided number of steps.
        /// Precondition: The provided number of steps must not be negative.
        /// </summary>
        /// <param name="theNumberOfRotations"></param>

        public void Rotate(int theNumberOfSteps)
        {
            this.indexOfTheFirstElement = (this.indexOfTheFirstElement + theNumberOfSteps) % this.array.Length;
        }


        /// <summary>
        /// setsItsElementGiven sets the element of this array rotator and provider at a
        /// provided index to a provided value.
        /// Precondition: The provided index must not be negative.
        /// </summary>
        /// <param name="theIndex"></param>
        /// <param name="theValue"></param>

        public void setsItsElementGiven(int theIndex, T theValue)
        {
            this.array[theIndex] = theValue;
        }

    }

}

package Com.TSL.UsBillsAndCoinsDeterminer;


import java.util.InputMismatchException;
import java.util.NoSuchElementException;
import java.util.Scanner;
import org.apache.commons.math3.util.Precision;


/** ********************************************************************************************************************
 * UsBillsAndCoinsDeterminer encapsulates the entry point to this program, which, given a United-States monetary amount,
 * determines the numbers of each United-States bill and coin needed to represent that amount with the smallest total
 * number of bills and coins. Types of United-States bills and coins include ten-dollar bill, five-dollar bill,
 * one-dollar bill, quarter, dime, nickel, and penny.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 05/31/21
 ******************************************************************************************************************** */

public class AUsBillsAndCoinsDeterminer 
{

    /** --------------------------------------------------------------------------------------------------------------
	 * main is the entry point to this program, which, given a United-States monetary amount, determines the numbers
	 * of each United-States bill and coin needed to represent that amount with the smallest total number of bills and
	 * coins.
	 * 
	 * @param args
	 -------------------------------------------------------------------------------------------------------------- */

    public static void main (String[] args)
        throws ACalculatedUsMonetaryAmountIsNotEqualToProvidedAmountException, AnInvalidUsMonetaryAmountException
    {
        
        AnInputAndOutputManager theInputAndOutputManager = new AnInputAndOutputManager ();
        double theUsMonetaryAmount = theInputAndOutputManager.requestsAUsMonetaryAmount ();

        AnAccountOfUsBillsAndCoins theAccountOfUsBillsAndCoins =
            new AnAccountOfUsBillsAndCoins (theUsMonetaryAmount);
        
        theInputAndOutputManager.printsInformationAbout (theAccountOfUsBillsAndCoins);
        
    }
    
}


/** *******************************************************************************************************************
 * ACalculatedUsMonetaryAmountIsNotEqualToProvidedAmountException represents the structure for an exception that occurs
 * when a calculated United-States monetary amount is not equal to the corresponding provided amount.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 05/31/21
 ******************************************************************************************************************* */

class ACalculatedUsMonetaryAmountIsNotEqualToProvidedAmountException extends Exception
{
    
    /** -------------------------------------------------------------------------------------------------------------
     * ACalculatedUsMonetaryAmountIsNotEqualToProvidedAmountException (String theMessage) is the one-parameter
     * constructor for ACalculatedUsMonetaryAmountIsNotEqualToProvidedAmountException. This constructor passes the
     * argument represented by its parameter theMessage to the one-parameter constructor of Exception.
     * 
     * @param theMessage
     ------------------------------------------------------------------------------------------------------------- */
    
    ACalculatedUsMonetaryAmountIsNotEqualToProvidedAmountException (String theMessage)
    {
        super (theMessage);
    }
    
}


/** ********************************************************************************************************************
 * AnInvalidMonetaryAmountException represents the structure for an exception that occurs when the program receives from
 * the user via the standard input stream an invalid monetary amount.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 05/31/21
 ******************************************************************************************************************** */

class AnInvalidUsMonetaryAmountException extends Exception
{
    
    /** -------------------------------------------------------------------------------------------------------------
     * AnInvalidMonetaryAmountException (String theMessage) is the one-parameter constructor for
     * AnInvalidMonetaryAmountException. This constructor passes the argument represented by its parameter theMessage
     * to the one-parameter constructor of Exception.
     * 
     * @param theMessage
     ------------------------------------------------------------------------------------------------------------- */
    
    AnInvalidUsMonetaryAmountException (String theMessage)
    {
        super (theMessage);
    }
    
}


/** ********************************************************************************************************************
 * ANoSuchElementException represents the structure for an exception that occurs when the program terminates after input
 * is requested from the user via the standard input stream and before input is received.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 05/31/21
 ******************************************************************************************************************** */

class ANoSuchElementException extends Exception
{
    
    /** -------------------------------------------------------------------------------------------------------------
     * ANoSuchElementException (String theMessage) is the one-parameter constructor for ANoSuchElementException. This
     * constructor passes the argument represented by its parameter theMessage to the one-parameter constructor of
     * Exception.
     * 
     * @param theMessage
     ------------------------------------------------------------------------------------------------------------- */
    
    ANoSuchElementException (String theMessage)
    {
        super (theMessage);
    }
    
}


/** ***********************************************************************************************************
 * AnAccountOfBillsAndCoins represents the structure for an account of bills and coins based on a United-States
 * monetary amount.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/01/31
 *********************************************************************************************************** */

class AnAccountOfUsBillsAndCoins
{
    
    /** --------------------------------------------------------------------------------------------------------
     * originalUsMonetaryAmount, numberOfTenDollarBills, numberOfFiveDollarBills, numberOfOneDollarBills,
     * numberOfQuarters, numberOfDimes, numberOfNickels, and numberOfPennies are all components of
     * AnAccountOfUsBillsAndCoins.
     -------------------------------------------------------------------------------------------------------- */
    
    private double providedUsMonetaryAmount;
    
    private double numberOfTenDollarBills;
    private double numberOfFiveDollarBills;
    private double numberOfOneDollarBills;
    private double numberOfQuarters;
    private double numberOfDimes;
    private double numberOfNickels;
    private double numberOfPennies;
    private double remainder;
    
    
    /** ---------------------------------------------------------------------------------------------------------
     * AnAccountOfUsBillsAndCoins (double theMonetaryAmount) is the one-parameter constructor for
     * AnAccountOfUsBillsAndCoins, which sets the components of this account based on the argument represented by
     * the parameter theUsMonetaryAmount.
     * 
     * @param theUsMonetaryAmount
     --------------------------------------------------------------------------------------------------------- */
    
    AnAccountOfUsBillsAndCoins (double theUsMonetaryAmount)
        throws ACalculatedUsMonetaryAmountIsNotEqualToProvidedAmountException
    {
        this.providedUsMonetaryAmount = theUsMonetaryAmount;
        
        this.numberOfTenDollarBills =
            Precision.round (theUsMonetaryAmount / 10.0, 0, ARoundingMethod.ROUND_DOWN.ordinal ());

        this.remainder = theUsMonetaryAmount - this.numberOfTenDollarBills * 10.0;
        
        this.numberOfFiveDollarBills = Precision.round (this.remainder / 5.0, 0, ARoundingMethod.ROUND_DOWN.ordinal ());
        
        this.remainder -= this.numberOfFiveDollarBills * 5.0;
        
        this.numberOfOneDollarBills = Precision.round (this.remainder / 1.0, 0, ARoundingMethod.ROUND_DOWN.ordinal ());
        
        this.remainder -= this.numberOfOneDollarBills * 1.0;
        
        this.numberOfQuarters = Precision.round (this.remainder / 0.25, 0, ARoundingMethod.ROUND_DOWN.ordinal ());
        
        this.remainder -= this.numberOfQuarters * 0.25;
        
        this.numberOfDimes = Precision.round (this.remainder / 0.10, 0, ARoundingMethod.ROUND_DOWN.ordinal ());
        
        this.remainder -= this.numberOfDimes * 0.10;
        
        this.numberOfNickels = Precision.round (this.remainder / 0.05, 0, ARoundingMethod.ROUND_DOWN.ordinal ());
        
        this.remainder -= this.numberOfNickels * 0.05;
        
        this.numberOfPennies = Precision.round (this.remainder / 0.01, 0, ARoundingMethod.ROUND_HALF_UP.ordinal ());
        
        this.remainder -= this.numberOfPennies * 0.01;
        
        compareACalculatedUsMonetaryAmountTo (this.providedUsMonetaryAmount);
        
    }
    
    
    /** -------------------------------------------------------------------------------------------------------------
     * compareACalculatedUsMonetaryAmountTo compares a United-States monetary amount calculated from numbers of bills
     * and coins to a provided amount and throws an exception if the amounts are not equal.
     * 
     * @param theOriginalUsMonetaryAmount
     * @throws ACalculatedUsMonetaryAmountIsNotEqualToProvidedAmountException
     ------------------------------------------------------------------------------------------------------------ */
    
    void compareACalculatedUsMonetaryAmountTo (double theOriginalUsMonetaryAmount)
        throws ACalculatedUsMonetaryAmountIsNotEqualToProvidedAmountException
    {
        double theCalculatedUsMonetaryAmount =
            Precision.round (
                this.numberOfTenDollarBills * 10.0 +
                this.numberOfFiveDollarBills * 5.0 +
                this.numberOfOneDollarBills * 1.0 +
                this.numberOfQuarters * 0.25 +
                this.numberOfDimes * 0.10 +
                this.numberOfNickels * 0.05 +
                this.numberOfPennies * 0.01
                , 2
           );
        
        if (theCalculatedUsMonetaryAmount != this.providedUsMonetaryAmount)
        {
            throw new ACalculatedUsMonetaryAmountIsNotEqualToProvidedAmountException ("Exception");
        }
    }
    
    
    /** --------------------------------------------------------------------------------------------------
     * givesItsProvidedMonetaryAmount gives this account's provided monetary amount to the calling method.
     * 
     * @return
     -------------------------------------------------------------------------------------------------- */
    
    double givesItsProvidedMonetaryAmount()
    {
        return this.providedUsMonetaryAmount;
    }
    
    
    /** -----------------------------------------------------------
     * providesItsRemainder provides the remainder of this account.
     * 
     * @return
     ----------------------------------------------------------- */
    
    double providesItsRemainder()
    {
        return this.remainder;
    }
    
    
    
    /** --------------------------------------------------
     * toString provides a representation of this account.
     -------------------------------------------------- */
    
    @Override
    public String toString()
    {
        
        return
            String.format ("%.0f", this.numberOfTenDollarBills) + " ten dollar bills\n" +
            String.format ("%.0f", this.numberOfFiveDollarBills) + " five dollar bills\n" +
            String.format ("%.0f", this.numberOfOneDollarBills) + " one dollar bills\n" +
            String.format ("%.0f", this.numberOfQuarters) + " quarters\n" +
            String.format ("%.0f", this.numberOfDimes) + " dimes\n" +
            String.format ("%.0f", this.numberOfNickels) + " nickels\n" +
            String.format ("%.0f", this.numberOfPennies) + " pennies";
        
    }
    
}


/** **************************************************************
 * AnInputManager manages input of United-States monetary amounts.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 05/31/21
 ************************************************************** */

class AnInputAndOutputManager
{    
    
    /** --------------------------------------------------------------------------------------------------------
	 * requestsAndProvidesAUsMonetaryAmount requests a valid United-States monetary amount from the user via the
	 * standard input stream, and provides the monetary amount to the calling method.
	 * 
	 * TODO: Ask Chris Dilbeck how to introduce looping as long as necessary to provide a valid amount.
	 * 
	 * @return
	 -------------------------------------------------------------------------------------------------------- */
    
    double requestsAUsMonetaryAmount() throws AnInvalidUsMonetaryAmountException
    {
        
        System.out.print (
            "This program, given a United-States monetary amount, determines the numbers of each United-States bill " +
            "and coin needed to\nrepresent that amount with the smallest total number of bills and coins.\n\n" +
            
            "Please enter a United-States monetary amount, without '$': "
        );
        
        double theProposedUsMonetaryAmount = 0.0;
        Scanner theScannerOfTheStandardInputStream = new Scanner (System.in);
        
        try
        {
            // nextDouble throws an input mismatch exception if the next double does not match the Double regular
            // expression or the double is out of range. nextDouble throws a NoSuchElementException if the program
            // is terminated after input is requested and before input is received. nextDouble throws an
            // IllegalStateException if the scanner is closed when nextDouble is called.
            theProposedUsMonetaryAmount = theScannerOfTheStandardInputStream.nextDouble ();
            theScannerOfTheStandardInputStream.close ();
            
            // check throws AnInvalidMonetaryAmountException if the proposed monetary amount is invalid.
            check (theProposedUsMonetaryAmount);
        }
        catch (InputMismatchException theInputMismatchException)
        {
            theScannerOfTheStandardInputStream.close ();
            throw new InputMismatchException ("Input Mismatch Exception: The program received a non-double value.");
        }
        catch (NoSuchElementException theNoSuchElementException)
        {
            System.out.println (
                "No Such Element Exception: The program was terminated after input was requested from the user and " +
                "before input was received."
            );
        }
        catch (IllegalStateException theIllegalStateException)
        {
            throw new IllegalStateException (
                "Illegal State Exception: Scanner was closed before nextDouble was called."
            );
        }
            
        return theProposedUsMonetaryAmount;
    
    }
    
    
    /** -----------------------------------------------------------------------------------------------------------
     * printsInformation about prints information about a United-States monetary amount and an account of bills and
     * coins.
     * 
     * @param theUsMonetaryAmount
     * @param theAccountOfUsBillsAndCoins
     ----------------------------------------------------------------------------------------------------------- */
    
    void printsInformationAbout (AnAccountOfUsBillsAndCoins theAccountOfUsBillsAndCoins)
    {
        double theProvidedUsMonetaryAmount = theAccountOfUsBillsAndCoins.givesItsProvidedMonetaryAmount ();
        double theRemainder = theAccountOfUsBillsAndCoins.providesItsRemainder ();
        
        System.out.println (
            "\nBelow are the numbers of each United-States bill and coin needed to represent the United-States " +
            "monetary amount " + theProvidedUsMonetaryAmount + " with the\nsmallest total number of bills and coins.\n"
        );
        
        System.out.println (theAccountOfUsBillsAndCoins + "\n");
        
        System.out.println (
            "There exists a remainder of " + theRemainder + " since the decimal United-States monetary amount " +
            theProvidedUsMonetaryAmount + ( (theRemainder == 0.0) ? " can" : " cannot") + " be represented\nexactly " +
            "using the binary number system."
        );
        
    }
    
    
    /** -------------------------------------------------------------------------
     * check evaluates whether a proposed United-States monetary amount is valid.
     * 
     * @param theProposedUsMonetaryAmount
     * @throws AnInvalidMonetaryAmountException
     ------------------------------------------------------------------------- */
    
    private static void check (double theProposedUsMonetaryAmount) throws AnInvalidUsMonetaryAmountException
    {   

        if (theProposedUsMonetaryAmount != Precision.round (theProposedUsMonetaryAmount, 2))
        {
            throw new AnInvalidUsMonetaryAmountException ("Exception: The program received an invalid monetary amount.");
        }
        
    }

}


/** ******************************************************************************************************************
 * ARoundingMethod enumerates various rounding methods. See information about the rounding-method fields of BigDecimal
 * at https://docs.oracle.com/javase/8/docs/api/java/math/BigDecimal.html.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 05/31/21
 ***************************************************************************************************************** */

enum ARoundingMethod
{
    ROUND_UP,
    ROUND_DOWN,
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_UP,
    ROUND_HALF_DOWN,
    ROUND_HALF_EVEN,
    ROUND_UNNECESSARY
}
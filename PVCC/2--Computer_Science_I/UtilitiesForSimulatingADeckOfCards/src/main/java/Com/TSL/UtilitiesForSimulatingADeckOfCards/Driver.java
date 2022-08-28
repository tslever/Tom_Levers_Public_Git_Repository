package Com.TSL.UtilitiesForSimulatingADeckOfCards;


/** ****************************************************************************************************************
 * Driver encapsulates the entry point of this program, which creates a deck of cards, deals all cards, shuffles the
 * deck, and deals all cards.
 *
 * @author Tom Lever
 * @version 1.0
 * @since 07/13/21
 *************************************************************************************************************** */

public class Driver 
{
    
    /** --------------------------------------------------------------------------------------------------------------
     * main is the entry point of this program, which creates a deck of cards, deals all cards, shuffles the deck, and
     * deals all cards.
     * 
     * @param args
     * @throws ADealFromAnEmptyDeckException
     -------------------------------------------------------------------------------------------------------------- */
    
    public static void main(String[] args) throws ADealFromAnEmptyDeckException
    {
        DeckOfCards theDeckOfCards = new DeckOfCards();
        
        System.out.println("The following statements describe dealing all cards from a standard deck of cards.");
        while (!theDeckOfCards.isEmpty())
        {
            System.out.println("There are " + theDeckOfCards.getTheNumberOfCards() + " cards left in the deck.");
            theDeckOfCards.dealACard();
        }
        System.out.println("There are " + theDeckOfCards.getTheNumberOfCards() + " cards left in the deck.");
        System.out.println();
        
        theDeckOfCards.shuffle();
        
        System.out.println(
            "The following statements describe dealing all cards from a shuffled standard deck of cards."
        );
        while (!theDeckOfCards.isEmpty())
        {
            theDeckOfCards.dealACard();
        }
    }
    
}

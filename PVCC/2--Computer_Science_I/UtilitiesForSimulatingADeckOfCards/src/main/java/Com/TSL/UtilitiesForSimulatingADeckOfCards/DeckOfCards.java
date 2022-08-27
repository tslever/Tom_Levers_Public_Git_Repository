package Com.TSL.UtilitiesForSimulatingADeckOfCards;


import org.apache.commons.lang3.ArrayUtils;


/** ********************************************************
 * DeckOfCards represents the structure for a deck of cards.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 07/13/21
 ******************************************************** */

public class DeckOfCards
{

    private Card[] arrayOfCards;
    private int numberOfCards;
    
    
    /** ----------------------------------------------------------------------------------------------------------
     * DeckOfCards() is the zero-parameter constructor for DeckOfCards, which sets this deck to its default state.
     ---------------------------------------------------------------------------------------------------------- */
    
    public DeckOfCards()
    {
        setToTheDefaultState();
    }
    
    
    /** -------------------------------------
     * dealACard deals a card from this deck.
     * 
     * @return
     * @throws ADealFromAnEmptyDeckException
     ------------------------------------- */
    
    public Card dealACard() throws ADealFromAnEmptyDeckException
    {
        if (isEmpty())
        {
            throw new ADealFromAnEmptyDeckException("A deal from an empty deck was requested.");
        }
        
        this.numberOfCards--;
        
        Card theCardToDeal = this.arrayOfCards[this.numberOfCards];
        
        System.out.println("The card \"" + theCardToDeal + "\" is being dealt.");
        
        return theCardToDeal;
    }
    
    
    /** ----------------------------------------------------------------------
     * getTheNumberOfCardsInTheDeck provides the number of cards in this deck.
     * 
     * @return
     ---------------------------------------------------------------------- */
    
    public int getTheNumberOfCards()
    {
        return this.numberOfCards;
    }
    
    
    /** ---------------------------------------------------
     * isEmpty indicates whether or not this deck is empty.
     * 
     * @return
     --------------------------------------------------- */
    
    public boolean isEmpty()
    {
        return (this.numberOfCards == 0);
    }
    
    
    /** ------------------------------------------------------
     * setToTheDefaultState sets to a default state this deck.
     ------------------------------------------------------ */
    
    private void setToTheDefaultState()
    {
        final int THE_MAXIMUM_NUMBER_OF_CARDS_IN_A_DECK = 52;
        final int NUM_FACES = 13;
        final int NUM_SUITS = 4;
        
        this.arrayOfCards = new Card[THE_MAXIMUM_NUMBER_OF_CARDS_IN_A_DECK];
        
        int j;
        for (int i = 0; i < NUM_FACES; i++)
        {
            for (j = 0; j < NUM_SUITS; j++)
            {
                this.arrayOfCards[NUM_SUITS * i + j] = new Card(i + 1, j + 1);
            }
        }
        
        this.numberOfCards = THE_MAXIMUM_NUMBER_OF_CARDS_IN_A_DECK;
    }
    
    
    /** -----------------------------------------------------------------------------------------------
     * shuffle sets this deck back to its default state with no cards drawn, then randomizes this deck.
     ----------------------------------------------------------------------------------------------- */
    
    public void shuffle()
    {
        setToTheDefaultState();
        
        ArrayUtils.shuffle(this.arrayOfCards);
    }

}
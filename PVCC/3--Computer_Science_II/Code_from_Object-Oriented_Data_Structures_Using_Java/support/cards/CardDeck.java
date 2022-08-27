//----------------------------------------------------------------------
// CardDeck.java           by Dale/Joyce/Weems                 Chapter 6
//
// Models a deck of cards. Includes shuffling and dealing.
//----------------------------------------------------------------------

package support.cards;

import java.util.Random;
import ch06.lists.ABList;
import java.util.Iterator;
import javax.swing.ImageIcon;

public class CardDeck
{
  public static final int NUMCARDS = 52;
  
  protected ABList<Card> deck;
  protected Iterator<Card> deal;
  
  public CardDeck()
  {
    deck = new ABList<Card>(NUMCARDS);
    ImageIcon image;
    for (Card.Suit suit : Card.Suit.values())
       for (Card.Rank rank : Card.Rank.values())
       {
         image = new ImageIcon("support/cards/" + suit + "_" + rank + "_RA.gif");     
         deck.add(new Card(rank, suit, image));
       }
    deal = deck.iterator();
  }

  public void shuffle()
  // Randomizes the order of the cards in the deck.
  // Resets the current deal.
  {
    Random rand = new Random(); // to generate random numbers 
    int randLoc;                // random location in card deck
    Card temp;                  // for swap of cards
    
    for (int i = (NUMCARDS - 1); i > 0; i--)
    {
      randLoc = rand.nextInt(i);  // random integer between 0 and i - 1
      temp = deck.get(randLoc);
      deck.set(randLoc, deck.get(i));
      deck.set(i, temp);
    }
    
    deal = deck.iterator();
  }
  
  public boolean hasNextCard()
  // Returns true if there are still cards left to be dealt; 
  // otherwise, returns false.
  {
    return (deal.hasNext());
  }
  
  public Card nextCard()
  // Precondition:  this.hasNextCard() == true
  //
  // Returns the next card for the current 'deal'.
  {
    return deal.next();
  }
}
 
//---------------------------------------------------------------------
// Pairs.java            by Dale/Joyce/Weems                  Chapter 6
//
// Simulates dealing poker hands to calculate the probability of 
// getting at least one pair of matching cards.
//----------------------------------------------------------------------

package ch06.apps;

import ch06.lists.*;
import support.cards.*;      // Card and CardDeck

public class Pairs 
{
  public static void main(String[] args)
  {
    final int HANDSIZE = 5;        // number of cards per hand
    final int NUMHANDS = 1000000;  // total number of hands
    int numPairs = 0;              // number of hands with pairs
    boolean isPair;                // status of current hand
    float probability;             // calculated probability

    Card card;                       // playing card
    CardDeck deck = new CardDeck();  // deck of playing cards
    
    ListInterface<Card> hand = new ABList<Card>(HANDSIZE); // the poker hand

    for (int i = 0; i < NUMHANDS; i++)
    {
      deck.shuffle();
      hand = new ABList<Card>(HANDSIZE);
      isPair = false;
      for (int j = 0; j < HANDSIZE; j++)
      {
        card = deck.nextCard();
        if (hand.contains(card))
          isPair = true;
        hand.add(card);
      }
      if (isPair)
        numPairs = numPairs + 1;
    }

    probability = numPairs/(float)NUMHANDS;

    System.out.println();
    System.out.print("There were " + numPairs + " hands out of " + NUMHANDS);
    System.out.println(" that had at least one pair of matched cards.");
    System.out.print("The probability of getting at least one pair,");
    System.out.print(" based on this simulation, is ");
    System.out.println(probability);
  }
}
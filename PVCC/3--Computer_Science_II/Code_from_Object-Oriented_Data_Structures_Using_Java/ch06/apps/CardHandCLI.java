//---------------------------------------------------------------------
// CardHandCLI.java        by Dale/Joyce/Weems                Chapter 6
//
// Allows user to organize a hand of playing cards.
// Uses a command line interface.
//----------------------------------------------------------------------
package ch06.apps;

import java.util.Scanner;
import ch06.lists.*;
import support.cards.*;       

public class CardHandCLI
{
  public static void main(String[] args)
  {
    Scanner scan = new Scanner(System.in);
    final int HANDSIZE = 5;
    int slot;

    Card card;                       // playing card
    CardDeck deck = new CardDeck();  // deck of playing cards
    
    ListInterface<Card> hand = new ABList<Card>(HANDSIZE); // user's hand

    deck.shuffle();
    hand.add(deck.nextCard());  // deals 1st card and places into hand

    for (int i = 1; i < HANDSIZE; i++)
    {
      System.out.println("\nYour hand so far:");
      slot = 0;
      for (Card c: hand)
      {
        System.out.println(slot + "\n  " + c);
        slot++;
      }
      System.out.println(slot);
      
      card = deck.nextCard();      
      System.out.print("Slot between 0 and " + i + " to put " + card + " > ");
      slot = scan.nextInt();
      hand.add(slot, card);
    }

    System.out.println("\nYour final hand is:");
    for (Card c: hand)
       System.out.println("  " + c);
  }
}
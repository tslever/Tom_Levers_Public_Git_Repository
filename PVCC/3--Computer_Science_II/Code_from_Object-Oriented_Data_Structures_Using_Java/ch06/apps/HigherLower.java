//--------------------------------------------------------------------
// HigherLower.java         by Dale/Joyce/Weems              Chapter 6
//
// Plays Higher or Lower? with user through command line interface. 
//--------------------------------------------------------------------
package ch06.apps;

import support.cards.*;          // Card, CardDeck
import java.util.Scanner;

public class HigherLower 
{
  public static void main(String[] args)
  {
    Scanner scan = new Scanner(System.in);
    char reply;
    
    Card card1, card2;
    CardDeck deck = new CardDeck();
    deck.shuffle(); 
        
    System.out.println("Welcome to \"Higher or Lower\". Good luck!");

    // First card
    card1 = deck.nextCard(); 
    System.out.println("\nFirst Card: " + card1);

    // Get prediction
    System.out.print("Higher (H) or Lower (L)? > ");
    reply = scan.nextLine().charAt(0);

    // Second card    
    card2 = deck.nextCard();
    System.out.println("\nSecond Card: " + card2);

    // Determine and display results    
    if ((card2.compareTo(card1) > 0) && (reply == 'H'))
      System.out.println("Correct");
    else
    if ((card2.compareTo(card1) < 0) && (reply == 'L'))
      System.out.println("Correct");
    else
      System.out.println("Incorrect");
  }
}
//----------------------------------------------------------------------
// Card.java              by Dale/Joyce/Weems                  Chapter 6
//
// Supports playing card objects having a suit, a rank, and an image.
// Only rank is used when comparing cards. Ace is "high".
//----------------------------------------------------------------------
package support.cards;

import javax.swing.ImageIcon;

public class Card implements Comparable<Card>
{
  public enum Rank {Two, Three, Four, Five, Six, Seven, Eight, Nine, 
                    Ten, Jack, Queen, King, Ace}

  public enum Suit {Club, Diamond, Heart, Spade}

  protected final Rank rank;
  protected final Suit suit;
  protected ImageIcon image;
    
  Card(Rank rank, Suit suit, ImageIcon image) 
  {
    this.rank = rank; this.suit = suit; this.image = image;
  }

  public Rank getRank() { return rank; }
  public Suit getSuit() { return suit; }
  public ImageIcon getImage() {return image;}

  @Override 
  public boolean equals(Object obj)
  // Returns true if 'obj' is a Card with same rank  
  // as this Card, otherwise returns false.
  {
     if (obj == this)
        return true;
     else 
     if (obj == null || obj.getClass() != this.getClass())
        return false;
     else
     {
        Card c = (Card) obj; 
        return (this.rank == c.rank);
     }
  }

  public int compareTo(Card other)
  // Compares this Card with 'other' for order. Returns a 
  // negative integer, zero, or a positive integer as this object 
  // is less than, equal to, or greater than 'other'.
  {
    return this.rank.compareTo(other.rank);
  }

  @Override
  public String toString() { return suit + " " + rank; }
}
 
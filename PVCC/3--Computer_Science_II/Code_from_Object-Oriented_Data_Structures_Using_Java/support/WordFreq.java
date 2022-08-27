//---------------------------------------------------------------------------
// WordFreq.java              by Dale/Joyce/Weems                   Chapter 7
//
// Defines word-frequency pairs
//---------------------------------------------------------------------------
package support;

import java.text.DecimalFormat;

public class WordFreq implements Comparable<WordFreq>
{
  private String word;
  private int freq;

  DecimalFormat fmt = new DecimalFormat("00000");

  public WordFreq(String newWord)
  {
    word = newWord;
    freq = 0;
   }

  public String getWordIs(){return word;}
  public int getFreq(){return freq;}

  public void inc()
  {
    freq++;
  }

  public int compareTo(WordFreq other)
  {
    return this.word.compareTo(other.word); 
  }

  public String toString()
  {
    return(fmt.format(freq) + " " + word);
  }
}
 
package Com.TSL.UtilitiesForCalculatingVocabularyDensityAndWordFrequencies;


/**
* @author YINGJIN CUI
* @version 1.0
* since   2020-03
*
*  Word.java:  Word class
*/

public class Word implements Comparable<Word>{

	
   private String word;
   private int frequency;

   
   /**
    * Word is the one-parameter constructor of Word, which sets the string word of this word object to a provided string
    * word, and sets the frequency of this word to 1.
    * 
    * @param word
    */
   
   public Word(String word) {
      this.word = word; 
      frequency=1; 
   }

   
   /**
    * getWord provides the string word of this word object.
    * 
    * @return
    */
   
   public String getWord(){
      return word;
   }
   
   
   /**
    * getFrequency provides the frequency of this word object.
    * @return
    */
   
   public int getFrequency(){
      return frequency;
   }
   
   
   /**
    * increaseFrequency increments the frequency of this word object.
    */
   
   public void increaseFrequency() {
      frequency++;
   }
   
   
   /**
    * compareTo provides the difference in frequency between this word and a provided word.
    */
   
   public int compareTo(Word w){
      //sort object based on frequency but in descending order;
      return w.getFrequency() - frequency;
   }
   
   
   /**
    * equals indicates whether or not the string word of this word object is equal to the string word of a provided
    * word object, ignoring case.
    */
   
   public boolean equals(Object w){
      return word.equalsIgnoreCase(((Word)w).getWord());
   }
   
   
   /**
    * toString provides a representation of this word object, including its string word and frequency.
    */
   
   public String toString(){
      return String.format("%s[%d]",word, frequency);
   }
}
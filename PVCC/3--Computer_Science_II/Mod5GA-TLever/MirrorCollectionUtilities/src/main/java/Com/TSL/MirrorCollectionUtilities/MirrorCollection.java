package Com.TSL.MirrorCollectionUtilities;


/**
* @author YINGJIN CUI
* @version 1.0
* since   2020-03
*
* Student name:  Tom Lever
* Completion date: 06/15/21
*
* MirrorCollection encapsulates the entry point of this program, and a mirrorCollection method. On entry, method main
* defines a reference of type CollectionInterface<Integer> to a collection of some integers. main adds integers to the
* collection and displays the collection. The main method of this class defines another reference of type
* CollectionInterface<Integer>, as a reference to the mirror image of the original collection, provided by the
* mirrorCollection method of an object of this class. The main method displays this mirror image.
*/

public class MirrorCollection<T>{

	
   /**
    * main is the entry point of this program. On entry, method main defines a reference of type
    * CollectionInterface<Integer> to a collection of some integers. main adds integers to the collection and displays
    * the collection. The main method of this class defines another reference of type CollectionInterface<Integer>, as a
    * reference to the mirror image of the original collection, provided by the mirrorCollection method of an object of
    * this class. The main method displays this mirror image.
    * 
    * @param args
    */
	
   public static void main(String[] args) {
	   
      CollectionInterface<Integer> col= new CollectionT<Integer>(5);
      
      //col.add(9);
      //col.add(3);
      //col.add(5);
      col.add(9);
      col.add(5);
      col.add(1);
      col.add(8);
      
      col.print();
      
      CollectionInterface<Integer> res = new MirrorCollection().mirrorCollection(col);
      res.print();
      
   }

   
   public CollectionInterface<T> mirrorCollection(CollectionInterface<T> col){
	   
      // *** Student task #1 ***  

      /* 
      Requirements: 
      This method creates and returns mirror image of the original collection of a parameterized type objects. 
      For example, if an Integer collection stores this sequence of values: [9, 5, 1, 8]
      Then its mirror image collection should store the following values:   [9, 5, 1, 8, 8, 1, 5, 9]

      *** Enter your code below *** 
      */

      CollectionT<T> theMirrorCollection = new CollectionT<T>(col.size() * 2);
      
      for (int i = 0; i < col.size(); i++) {
    	  theMirrorCollection.add(col.get(i));
      }
      for (int i = col.size() - 1; i >= 0; i--) {
    	  theMirrorCollection.add(col.get(i));
      }
      
      return theMirrorCollection;
	   
   }
   
}
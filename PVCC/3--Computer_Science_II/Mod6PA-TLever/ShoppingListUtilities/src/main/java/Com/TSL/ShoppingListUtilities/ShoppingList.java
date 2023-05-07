package Com.TSL.ShoppingListUtilities;


/**
* @author YINGJIN CUI
* @version 1.0
* since   2020-03
*
* Student name:  Tom Lever
* Completion date: 06/22/21
*
* ShoppingList represents the structure for a shopping list.
*/

public class ShoppingList {
	
	
   private Item[] list;
   private int numberOfItems;
   
   
   /**
    * ShoppingList(int initialSize) is the one-parameter constructor for ShoppingList, which sets the array of items
    * of this list to a new array of items with initial capacity initialCapacity.
    * 
    * @param initialSize
    */
   
   public ShoppingList(int initialCapacity) {
	   
      list = new Item[initialCapacity];
      
   }
   
   public void insert(Item item){

      // *** Student task #1 ***  

      /* 
	Requirements: 
 
      	1. if null item is not allowed: do nothing if null
      	2. if the list is full, double the array size-dynamic array allocation
      	3. No duplicated items-all items' names must be unique. If an item with the same name
      	   already exists in the list, simply add quantity to the existing item in the list
      	4. insert the item to the array to maintain sorted-items are sorted based on item names.

        *** Enter your code below *** 
     */

	   if (item == null) {
		   return;
	   }
	   
	   if (isFull()) {
		   doubling();		   
	   }
	   
	   int theIndexOfTheItem = indexOf(item);
	   if (theIndexOfTheItem > -1) {
		   Item theItemInTheList = this.list[theIndexOfTheItem];
		   theItemInTheList.setQuantity(theItemInTheList.getQuantity() + item.getQuantity());
		   return;
	   }
	   
	   int theIndexAtWhichToInsertTheItem = 0;
	   while ((theIndexAtWhichToInsertTheItem < size()) &&
			  (item.compareTo(this.list[theIndexAtWhichToInsertTheItem]) > 0)) {
		   theIndexAtWhichToInsertTheItem++;
	   }
	   
	   for (int i = size(); i > theIndexAtWhichToInsertTheItem; i--) {
		   this.list[i] = this.list[i - 1];
	   }
	   
	   this.list[theIndexAtWhichToInsertTheItem] = item;
	   
	   this.numberOfItems++;
      
   }
   
   
   
   public void remove(Item item) {

      // *** Student task #2 ***  

      /* 
	Requirements: 
  
      	1. If item not found(item name), display the item does not exists in the list
      	   otherwise, remove the item from the list.
      	2. Hint: You do need to shift all items after removed one.

        *** Enter your code below *** 
     */
	   
	   if (indexOf(item) == -1) {
		   System.out.println("the item does not exist in the list.");
		   return;
	   }
	   
	   for (int i = indexOf(item); i < this.size() - 1; i++) {
		   this.list[i] = this.list[i + 1];
	   }
	   
	   this.list[size() - 1] = null;
	   this.numberOfItems--;

   }
   
   public void remove(String name){		// remove item by its name

      // *** Student task #3 ***  

      /* 
	Requirements: 

      	1. If item not found(item name), display the item does not exists in the list
      	   otherwise, remove the item from the list.
      	2. Hint: You do need to shift all items after removed one.

        *** Enter your code below *** 
     */
	   
	   if (indexOf(name) == -1) {
		   System.out.println("the item does not exist in the list.");
		   return;
	   }
	   
	   for (int i = indexOf(name); i < this.size() - 1; i++) {
		   this.list[i] = this.list[i + 1];
	   }
	   
	   this.list[size() - 1] = null;
	   this.numberOfItems--;

   }
   
   private void doubling(){
      //Double the list size but keep all items in the list.
      Item[] tmp=new Item[list.length*2];
      for(int i=0; i<list.length; i++){
         tmp[i]=list[i];
      }
      list=tmp;
   }
   
   public int indexOf(Item item){
      // return the index of the item that has the same item name in the list array, return -1 if not found
      for(int i=0; i<numberOfItems; i++){
         if(list[i].compareTo(item)==0)
            return i;
      }
      return -1; // not found
   }

   public int indexOf(String name){
      // return the index of the item that has the same item name in the list array, return -1 if not found
      name=name.toLowerCase();
      for(int i=0; i<numberOfItems; i++){
         if(list[i].getName().toLowerCase().compareTo(name)==0)
            return i;
      }
      return -1; // not found
   }

   public boolean isFull(){
      return numberOfItems==list.length;
   }

   public boolean isEmpty(){
      return numberOfItems==0;
   }

   public int size(){
      return numberOfItems;
   }

   public void printNames(){
      System.out.print("[");
      for(int i=0; i<numberOfItems; i++){
         System.out.print(list[i].getName());
         if(i<numberOfItems-1)
            System.out.print(", ");
      }
      System.out.println("]");
   }
   
   public void print(){
      for(int i=0; i<numberOfItems; i++){
         System.out.println((i+1)+". "+list[i].toString());
      }
   }
}
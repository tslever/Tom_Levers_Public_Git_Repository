package Com.TSL.UtilitiesForShoppingListSortedByCategoryAndName;


/**
 * ShoppingListDriver encapsulates the entry point of this program, which tests creating a shopping list that will be
 * sorted by category and then by name, and tests inserting items into the list, outputting information about the list,
 * removing an item from the list, and outputting information about the list.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/24/21
 */

public class ShoppingListDriver {

	
	/**
	 * main is the entry point of this program, which tests creating a shopping list that will be sorted by category and
	 * then by name, and tests inserting items into the list, outputting information about the list, removing an item
	 * from the list, and outputting information about the list.
	 * 
	 * @param args
	 */
	
	public static void main(String[] args) {
		
      ShoppingList sl=new ShoppingList();
      
      sl.insert(null);
      sl.insert(new Item("Bread", "Carb Food", 2, 2.99));
      sl.insert(new Item("Seafood", "Sea Food", -1, 10.99));
      sl.insert(new Item("Rice", "Carb Food", 2, 19.99));
      sl.insert(new Item("Salad Dressings", "Dressing", 2, 19.99));
      sl.insert(new Item("Eggs", "Protein", 2, 3.99));
      sl.insert(new Item("Cheese","Protein", 2, 1.59));
      sl.insert(new Item("Eggs", "Protein", 3, 3.99));
      
      System.out.print("\nThe names of unique items in the shopping list: ");
      sl.printNames();
      System.out.println();
      sl.print();
      System.out.println();
      
      System.out.println("After removing Eggs:");
      sl.remove(new Item("Eggs", "Protein", 0, 0));
      System.out.print("\nThe names of unique items in the shopping list: ");
      sl.printNames();
      System.out.println();
      sl.print();
	
	}
	
}

package Com.TSL.UtilitiesForShoppingListSortedByCategoryAndName;


import org.apache.commons.math3.util.Precision;


/**
 * Item represents the structure of an item in a shopping list sorted first by category and then by name.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 06/23/21
 */

public class Item implements Comparable<Item> {

	
	private final int THE_MAXIMUM_LENGTH_OF_A_STRING = 50;
	
	
	/**
	 * Name represents the name of this item.
	 * The length of the string referenced by Name is limited to [1, 50].
	 * Neither a null reference nor an empty string may be used. 
	 */
	
	private String Name;
	
	
	/**
	 * Category represents the category of this item.
	 * The length of the string referenced by Category is limited to [1, 50].
	 * Neither a null reference nor an empty string may be used.
	 */
	
	private String Category;
	
	
	/**
	 * Quantity represents the quantity of this item.
	 * The value of Quantity is limited to [1, 100].
	 */
	
	private int Quantity;
	
	
	/**
	 * unitPrice represents the unit price of this item.
	 * The value of unitPrice is limited to [0.0, 1000.0].
	 */
	
	private double unitPrice;
	
	
	/**
	 * Item(String theNameToUse, String theCategoryToUse, int theQuantityToUse, double theUnitPriceToUse) is the
	 * four-parameter constructor for Item, which sets this item's name, category, quantity, and unit price based on
	 * appropriate provided values. If the provided name or category is null or empty, a runtime exception is thrown.
	 * If the provided name or category is longer than the maximum length of a string, the provided name or category
	 * is truncated. If the provided quantity is out of bounds, the provided quantity is adjusted to 1. If the
	 * provided unit price is not a United States monetary amount or is out of bounds, the provided unit price is
	 * adjusted.
	 */
	
	public Item(String theNameToUse, String theCategoryToUse, int theQuantityToUse, double theUnitPriceToUse) {
		
		setsItsNameTo(theNameToUse);
		setsItsCategoryTo(theCategoryToUse);
		setsItsQuantityTo(theQuantityToUse);
		setsItsUnitPriceTo(theUnitPriceToUse);
		
	}
	
	
	/**
	 * setsItsNameTo sets the name of this item.
	 */
	
	public void setsItsNameTo(String theNameToUse) {
		
		if (theNameToUse == null) {
			throw new RuntimeException("An item was constructed with reference theNameToUse as null.");
		}
		
		if (theNameToUse.equals("")) {
			throw new RuntimeException("An item was constructed with an empty name.");
		}
		
		if (theNameToUse.length() > THE_MAXIMUM_LENGTH_OF_A_STRING) {
			theNameToUse = theNameToUse.substring(0, THE_MAXIMUM_LENGTH_OF_A_STRING);
			System.out.println("Warning: The provided name was truncated to fifty characters.");
		}
		
		this.Name = theNameToUse;
		
	}
	
	
	/**
	 * setsItsCategoryTo sets the name of this item.
	 */
	
	public void setsItsCategoryTo(String theCategoryToUse) {
		
		if (theCategoryToUse == null) {
			throw new RuntimeException("An item was constructed with reference theCategoryToUse as null.");
		}
		
		if (theCategoryToUse.equals("")) {
			throw new RuntimeException("An item was constructed with an empty category.");
		}
		
		if (theCategoryToUse.length() > THE_MAXIMUM_LENGTH_OF_A_STRING) {
			theCategoryToUse = theCategoryToUse.substring(0, THE_MAXIMUM_LENGTH_OF_A_STRING);
			System.out.println("Warning: The provided name was truncated to fifty characters.");
		}
		
		this.Category = theCategoryToUse;
		
	}
	
	
	/**
	 * setsItsQuantityTo sets the quantity of this item.
	 * 
	 * @param theQuantityToUse
	 */
	
	public void setsItsQuantityTo(int theQuantityToUse) {
		
		if ((theQuantityToUse < 1) || (theQuantityToUse > 100)) {
			theQuantityToUse = 1;
			System.out.println("Warning: The provided quantity was outside of [1, 100] and was reset to 1.");
		}
		
		this.Quantity = theQuantityToUse;
		
	}
	
	
	/**
	 * setsItsUnitPriceTo the unit price of this item.
	 * 
	 * @param theUnitPriceToUse
	 */
	
	public void setsItsUnitPriceTo(double theUnitPriceToUse) {
		
		if (theUnitPriceToUse != Precision.round(theUnitPriceToUse, 2)) {
			theUnitPriceToUse = 0.99;
			System.out.println(
				"Warning: The provided unit price was not a U.S. monetary amount and was reset to 0.99."
			);
		}
		
		if ((theUnitPriceToUse < 0.0) || (theUnitPriceToUse > 1000.0)) {
			theUnitPriceToUse = 0.99;
			System.out.println("Warning: The provided unit price was outside of [0.0, 1000.0] and was reset to 0.99");
		}
		
		this.unitPrice = theUnitPriceToUse;
		
	}
	
	
	/**
	 * providesItsName provides the name of this item.
	 */
	
	public String providesItsName() {
		return this.Name;
	}
	
	
	/**
	 * providesItsCategory provides the category of this item.
	 * 
	 * @return
	 */
	
	public String providesItsCategory() {
		return this.Category;
	}
	
	
	/**
	 * providesItsQuantity provides the quantity of this item.
	 * 
	 * @return
	 */
	
	public int providesItsQuantity() {
		return this.Quantity;
	}
	
	
	/**
	 * providesItsUnitPrice provides the unit price of this item.
	 * 
	 * @return
	 */
	
	public double providesItsUnitPrice() {
		return this.unitPrice;
	}
	
	
	/**
	 * compareTo indicates whether this item is less than, equal to, or greater than an item to compare to this item.
	 */
	
	public int compareTo(Item theItemToCompare) {
		
		if (theItemToCompare == null) {
			throw new RuntimeException("The reference to the item to compare with an item was null.");
		}
		
		if (this.Category.compareToIgnoreCase(theItemToCompare.providesItsCategory()) < 0) {
			return -1;
		}
		
		if (this.Category.compareToIgnoreCase(theItemToCompare.providesItsCategory()) > 0) {
			return 1;
		}
		
		// At this point, the category of this item must be the same as the category of the item to compare to this
		// item.
		
		if (this.Name.compareToIgnoreCase(theItemToCompare.providesItsName()) < 0) {
			return -1;
		}
		
		if (this.Name.compareToIgnoreCase(theItemToCompare.providesItsName()) > 0) {
			return 1;
		}
		
		return 0;
		
	}
	
	
	/**
	 * Subtotal provides the subtotal for this item, which is the product of the quantity of this item and the unit
	 * price of this item.
	 * 
	 * @return
	 */
	
	public double Subtotal() {
		
		return Precision.round((double)this.Quantity * this.unitPrice, 2);
		
	}
	
	
	/**
	 * toString provides a string representation of this item.
	 */
	
	@Override
	public String toString() {
		
		return this.Name + "\t" + this.Quantity + "\t$" + Subtotal();
		
	}
	
	
	/**
	 * increasesItsQuantityBy increases the quantity of this item by a provided quantity.
	 * 
	 * @param theQuantityByWhichToIncrease
	 */
	
	public void increasesItsQuantityBy(int theQuantityByWhichToIncrease) {
		this.Quantity += theQuantityByWhichToIncrease;
	}
	
	
}

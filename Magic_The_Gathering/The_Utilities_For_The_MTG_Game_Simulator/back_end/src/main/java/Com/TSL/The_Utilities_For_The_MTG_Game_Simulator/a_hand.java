package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;

public class a_hand {

	private ArrayList<a_land_card> List_Of_Land_Cards;
	private ArrayList<a_nonland_card> List_Of_Nonland_Cards;
	
	public a_hand() {
		this.List_Of_Land_Cards = new ArrayList<a_land_card>();
		this.List_Of_Nonland_Cards = new ArrayList<a_nonland_card>();
	}
	
	public ArrayList<a_card> list_of_cards() {
		ArrayList<a_card> The_List_Of_Cards = new ArrayList<>();
		The_List_Of_Cards.addAll(this.List_Of_Land_Cards);
		The_List_Of_Cards.addAll(this.List_Of_Nonland_Cards);
		return The_List_Of_Cards;
	}
	
	public ArrayList<a_land_card> list_of_land_cards() {
		return this.List_Of_Land_Cards;
	}
	
	public ArrayList<a_nonland_card> list_of_nonland_cards() {
		return this.List_Of_Nonland_Cards;
	}
	
	public int number_of_cards() {
		return this.number_of_land_cards() + this.number_of_nonland_cards();
	}
	
	public int number_of_land_cards() {
		return this.List_Of_Land_Cards.size();
	}
	
	public int number_of_nonland_cards() {
		return this.List_Of_Nonland_Cards.size();
	}
	
	public void receives(a_card The_Card_To_Receive) {
		if (The_Card_To_Receive instanceof a_land_card) {
			this.List_Of_Land_Cards.add((a_land_card) The_Card_To_Receive);
		} else {
			this.List_Of_Nonland_Cards.add((a_nonland_card) The_Card_To_Receive);
		}
	}
	
	public void removes(a_card The_Card_To_Remove) {
		if (The_Card_To_Remove instanceof a_land_card) {
			a_land_card The_Land_Card_To_Remove = (a_land_card) The_Card_To_Remove; 
			this.List_Of_Land_Cards.remove(The_Land_Card_To_Remove);
		} else {
			a_nonland_card The_Nonland_Card_To_Remove = (a_nonland_card) The_Card_To_Remove;
			this.List_Of_Nonland_Cards.remove(The_Nonland_Card_To_Remove);
		}
	}
	
	@Override
	public String toString() {
		StringBuilder The_String_Builder = new StringBuilder("Hand: ");
		for (int i = 0; i < this.List_Of_Land_Cards.size() - 1; i++) {
			The_String_Builder.append(this.List_Of_Land_Cards.get(i));
			The_String_Builder.append("; ");
		}
		if (this.List_Of_Land_Cards.size() > 0) {
		    The_String_Builder.append(this.List_Of_Land_Cards.get(this.List_Of_Land_Cards.size() - 1));
		}
		if (!this.List_Of_Land_Cards.isEmpty() && !this.List_Of_Nonland_Cards.isEmpty()) {
			The_String_Builder.append("; ");
		}
		for (int i = 0; i < this.List_Of_Nonland_Cards.size() - 1; i++) {
			The_String_Builder.append(this.List_Of_Nonland_Cards.get(i));
			The_String_Builder.append("; ");
		}
		if (this.List_Of_Nonland_Cards.size() > 0) {
		    The_String_Builder.append(this.List_Of_Nonland_Cards.get(this.List_Of_Nonland_Cards.size() - 1));
		}
		return The_String_Builder.toString();
	}
}
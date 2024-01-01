package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;

public class a_hand {

	private ArrayList<a_land_card> List_Of_Land_Cards;
	private ArrayList<a_nonland_card> List_Of_Nonland_Cards;
	
	public a_hand() {
		this.List_Of_Land_Cards = new ArrayList<a_land_card>();
		this.List_Of_Nonland_Cards = new ArrayList<a_nonland_card>();
	}
	
	public ArrayList<a_card> provides_its_list_of_cards() {
		ArrayList<a_card> The_List_Of_Cards = new ArrayList<>();
		The_List_Of_Cards.addAll(this.List_Of_Land_Cards);
		The_List_Of_Cards.addAll(this.List_Of_Nonland_Cards);
		return The_List_Of_Cards;
	}
	
	public ArrayList<a_land_card> provides_its_list_of_land_cards() {
		return this.List_Of_Land_Cards;
	}
	
	public ArrayList<a_nonland_card> provides_its_list_of_nonland_cards() {
		return this.List_Of_Nonland_Cards;
	}
	
	public int provides_its_number_of_cards() {
		return this.provides_its_number_of_land_cards() + this.provides_its_number_of_nonland_cards();
	}
	
	public int provides_its_number_of_land_cards() {
		return this.List_Of_Land_Cards.size();
	}
	
	public int provides_its_number_of_nonland_cards() {
		return this.List_Of_Nonland_Cards.size();
	}
	
	public void receives(a_card The_Card_To_Receive) {
		if (The_Card_To_Receive instanceof a_land_card) {
			this.List_Of_Land_Cards.add((a_land_card) The_Card_To_Receive);
		} else {
			this.List_Of_Nonland_Cards.add((a_nonland_card) The_Card_To_Receive);
		}
	}
	
	public a_nonland_card plays(a_nonland_card The_Nonland_Card_To_Play) throws Exception {
		for (int i = 0; i < this.List_Of_Nonland_Cards.size(); i++) {
			a_nonland_card The_Nonland_Card = this.List_Of_Nonland_Cards.get(i);
			if (The_Nonland_Card.equals(The_Nonland_Card_To_Play)) {
				return this.List_Of_Nonland_Cards.remove(i);
			}
		}
		throw new Exception("The end of a hand.play was reached.");
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
		if (this.List_Of_Nonland_Cards.size() > 0) {
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
package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


import java.util.ArrayList;


public class a_hand {

	private ArrayList<an_artifact_card> List_Of_Artifact_Cards;
	private ArrayList<a_creature_card> List_Of_Creature_Cards;
	private ArrayList<an_enchantment_card> List_Of_Enchantment_Cards;
	private ArrayList<an_instant_card> List_Of_Instant_Cards;
	private ArrayList<a_land_card> List_Of_Land_Cards;
	private ArrayList<a_planeswalker_card> List_Of_Planeswalker_Cards;
	private ArrayList<a_sorcery_card> List_Of_Sorcery_Cards;
	
	
	public a_hand() {
		
		this.List_Of_Artifact_Cards = new ArrayList<an_artifact_card>();
		this.List_Of_Creature_Cards = new ArrayList<a_creature_card>();
		this.List_Of_Enchantment_Cards = new ArrayList<an_enchantment_card>();
		this.List_Of_Instant_Cards = new ArrayList<an_instant_card>();
		this.List_Of_Land_Cards = new ArrayList<a_land_card>();
		this.List_Of_Planeswalker_Cards = new ArrayList<a_planeswalker_card>();
		this.List_Of_Sorcery_Cards = new ArrayList<a_sorcery_card>();
		
	}
	
	public ArrayList<an_artifact_card> provides_its_list_of_artifact_cards() {
		return this.List_Of_Artifact_Cards;
	}
	
	public ArrayList<a_card> provides_its_list_of_cards() {
		
		ArrayList<a_card> The_List_Of_Cards = new ArrayList<a_card>();
		
		for (an_artifact_card The_Artifact_Card : this.List_Of_Artifact_Cards) {
			The_List_Of_Cards.add(The_Artifact_Card);
		}
		for (a_creature_card The_Creature_Card : this.List_Of_Creature_Cards) {
			The_List_Of_Cards.add(The_Creature_Card);
		}
		for (an_enchantment_card The_Enchantment_Card : this.List_Of_Enchantment_Cards) {
			The_List_Of_Cards.add(The_Enchantment_Card);
		}
		for (an_instant_card The_Instant_Card : this.List_Of_Instant_Cards) {
			The_List_Of_Cards.add(The_Instant_Card);
		}
		for (a_land_card The_Land_Card : this.List_Of_Land_Cards) {
			The_List_Of_Cards.add(The_Land_Card);
		}
		for (a_planeswalker_card The_Planeswalker_Card : this.List_Of_Planeswalker_Cards) {
			The_List_Of_Cards.add(The_Planeswalker_Card);
		}
		for (a_sorcery_card The_Sorcery_Card : this.List_Of_Sorcery_Cards) {
			The_List_Of_Cards.add(The_Sorcery_Card);
		}
		
		return The_List_Of_Cards;
	}
	
	public ArrayList<a_creature_card> provides_its_list_of_creature_cards() {
		return this.List_Of_Creature_Cards;
	}
	
	public ArrayList<an_enchantment_card> provides_its_list_of_enchantment_cards() {
		return this.List_Of_Enchantment_Cards;
	}
	
	public ArrayList<an_instant_card> provides_its_list_of_instant_cards() {
		return this.List_Of_Instant_Cards;
	}
	
	public ArrayList<a_land_card> provides_its_list_of_land_cards() {
		return this.List_Of_Land_Cards;
	}

	public ArrayList<a_card> provides_its_list_of_nonland_cards() {
		
		ArrayList<a_card> The_List_Of_Cards = new ArrayList<a_card>();
		
		for (an_artifact_card The_Artifact_Card : this.List_Of_Artifact_Cards) {
			The_List_Of_Cards.add(The_Artifact_Card);
		}
		for (a_creature_card The_Creature_Card : this.List_Of_Creature_Cards) {
			The_List_Of_Cards.add(The_Creature_Card);
		}
		for (an_enchantment_card The_Enchantment_Card : this.List_Of_Enchantment_Cards) {
			The_List_Of_Cards.add(The_Enchantment_Card);
		}
		for (an_instant_card The_Instant_Card : this.List_Of_Instant_Cards) {
			The_List_Of_Cards.add(The_Instant_Card);
		}
		for (a_planeswalker_card The_Planeswalker_Card : this.List_Of_Planeswalker_Cards) {
			The_List_Of_Cards.add(The_Planeswalker_Card);
		}
		for (a_sorcery_card The_Sorcery_Card : this.List_Of_Sorcery_Cards) {
			The_List_Of_Cards.add(The_Sorcery_Card);
		}
		
		return The_List_Of_Cards;
	}
	
	public ArrayList<a_planeswalker_card> provides_its_list_of_planeswalker_cards() {
		return this.List_Of_Planeswalker_Cards;
	}
	
	public ArrayList<a_sorcery_card> provides_its_list_of_sorcery_cards() {
		return this.List_Of_Sorcery_Cards;
	}
	
	
	public int provides_its_number_of_cards() {
		return
			this.List_Of_Artifact_Cards.size() +
			this.List_Of_Creature_Cards.size() +
			this.List_Of_Enchantment_Cards.size() +
			this.List_Of_Instant_Cards.size() +
			this.List_Of_Land_Cards.size() +
			this.List_Of_Planeswalker_Cards.size() +
			this.List_Of_Sorcery_Cards.size();
	}
	
	
	public int provides_its_number_of_land_cards() {
		return this.List_Of_Land_Cards.size();
	}
	
	
	public a_land_card provides_the_land_card_at_index(int The_Index_Of_The_Land_Card) {
		return this.List_Of_Land_Cards.remove(The_Index_Of_The_Land_Card);
	}
	
	
	public void receives(a_card The_Card) {
		switch (The_Card.provides_its_type()) {
			case "Artifact":
				this.List_Of_Artifact_Cards.add((an_artifact_card)The_Card);
				break;
			case "Basic Land":
				this.List_Of_Land_Cards.add((a_land_card)The_Card);
				break;
			case "Creature":
				this.List_Of_Creature_Cards.add((a_creature_card)The_Card);
				break;
			case "Enchantment":
				this.List_Of_Enchantment_Cards.add((an_enchantment_card)The_Card);
				break;
			case "Instant":
				this.List_Of_Instant_Cards.add((an_instant_card)The_Card);
				break;
			case "Planeswalker":
				this.List_Of_Planeswalker_Cards.add((a_planeswalker_card)The_Card);
				break;
			case "Sorcery":
				this.List_Of_Sorcery_Cards.add((a_sorcery_card)The_Card);
				break;
		}
	}
	
	
	@Override
	public String toString() {
		StringBuilder The_String_Builder = new StringBuilder("Hand");
		for (an_artifact_card The_Artifact_Card : this.List_Of_Artifact_Cards) {
			The_String_Builder.append("\n");
			The_String_Builder.append(The_Artifact_Card);
		}
		for (a_creature_card The_Creature_Card : this.List_Of_Creature_Cards) {
			The_String_Builder.append("\n");
			The_String_Builder.append(The_Creature_Card);
		}
		for (an_enchantment_card The_Enchantment_Card : this.List_Of_Enchantment_Cards) {
			The_String_Builder.append("\n");
			The_String_Builder.append(The_Enchantment_Card);
		}
		for (an_instant_card The_Instant_Card : this.List_Of_Instant_Cards) {
			The_String_Builder.append("\n");
			The_String_Builder.append(The_Instant_Card);
		}
		for (a_land_card The_Land_Card : this.List_Of_Land_Cards) {
			The_String_Builder.append("\n");
			The_String_Builder.append(The_Land_Card);
		}
		for (a_planeswalker_card The_Planeswalker_Card : this.List_Of_Planeswalker_Cards) {
			The_String_Builder.append("\n");
			The_String_Builder.append(The_Planeswalker_Card);
		}
		for (a_sorcery_card The_Sorcery_Card : this.List_Of_Sorcery_Cards) {
			The_String_Builder.append("\n");
			The_String_Builder.append(The_Sorcery_Card);
		}

		return The_String_Builder.toString();
	}
	
}

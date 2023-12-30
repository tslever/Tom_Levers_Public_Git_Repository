package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


import java.util.ArrayList;


public class a_part_of_the_battlefield {

	private ArrayList<an_artifact> List_Of_Artifacts;
	private ArrayList<a_creature> List_Of_Creatures;
	private ArrayList<an_enchantment> List_Of_Enchantments;
	private ArrayList<a_land> List_Of_Lands;
	private ArrayList<a_planeswalker> List_Of_Planeswalkers;
	
	
	public a_part_of_the_battlefield() {
		this.List_Of_Artifacts = new ArrayList<an_artifact>();
		this.List_Of_Creatures = new ArrayList<a_creature>();
		this.List_Of_Enchantments = new ArrayList<an_enchantment>();
		this.List_Of_Lands = new ArrayList<a_land>();
		this.List_Of_Planeswalkers = new ArrayList<a_planeswalker>();
	}
	
	
	public ArrayList<an_artifact> provides_its_list_of_artifacts() {
		return this.List_Of_Artifacts;
	}
	
	
	public ArrayList<a_creature> provides_its_list_of_creatures() {
		return this.List_Of_Creatures;
	}
	
	
	public ArrayList<an_enchantment> provides_its_list_of_enchantments() {
		return this.List_Of_Enchantments;
	}
	
	
	public ArrayList<a_land> provides_its_list_of_lands() {
		return this.List_Of_Lands;
	}
	
	
	public ArrayList<a_planeswalker> provides_its_list_of_planeswalkers() {
		return this.List_Of_Planeswalkers;
	}
	
	
	public ArrayList<a_permanent> provides_its_list_of_permanents() {
		
		ArrayList<a_permanent> The_List_Of_Permanents = new ArrayList<a_permanent>();
		
		for (an_artifact The_Artifact : this.List_Of_Artifacts) {
			The_List_Of_Permanents.add(The_Artifact);
		}
		for (a_creature The_Creature : this.List_Of_Creatures) {
			The_List_Of_Permanents.add(The_Creature);
		}
		for (an_enchantment The_Enchantment : this.List_Of_Enchantments) {
			The_List_Of_Permanents.add(The_Enchantment);
		}
		for (a_land The_Land : this.List_Of_Lands) {
			The_List_Of_Permanents.add(The_Land);
		}
		for (a_planeswalker The_Planeswalker : this.List_Of_Planeswalkers) {
			The_List_Of_Permanents.add(The_Planeswalker);
		}
		
		return The_List_Of_Permanents;
	}
	
	
	public int provides_its_number_of_permanents() {
		return
			this.List_Of_Artifacts.size() +
			this.List_Of_Creatures.size() +
			this.List_Of_Enchantments.size() +
			this.List_Of_Lands.size() +
			this.List_Of_Planeswalkers.size();
	}
	
	
	public void receives_creature(a_creature The_Creature) {
		this.List_Of_Creatures.add(The_Creature);
	}
	
	
	public void receives_land(a_land The_Land) {
		this.List_Of_Lands.add(The_Land);
	}
	
	
	@Override
	public String toString() {
		
		StringBuilder The_String_Builder = new StringBuilder("Part of the Battlefield: ");
		
		for (int i = 0; i < this.List_Of_Artifacts.size() - 1; i++) {
			The_String_Builder.append(this.List_Of_Artifacts.get(i));
			The_String_Builder.append("; ");
		}
		if (this.List_Of_Artifacts.size() > 0) {
			The_String_Builder.append(this.List_Of_Artifacts.get(this.List_Of_Artifacts.size() - 1));
		}
		
		if (this.List_Of_Artifacts.size() > 0 && this.List_Of_Creatures.size() > 0) {
			The_String_Builder.append("; ");
		}
		for (int i = 0; i < this.List_Of_Creatures.size() - 1; i++) {
			The_String_Builder.append(this.List_Of_Creatures.get(i));
			The_String_Builder.append("; ");
		}
		if (this.List_Of_Creatures.size() > 0) {
			The_String_Builder.append(this.List_Of_Creatures.get(this.List_Of_Creatures.size() - 1));
		}
		
		if ((this.List_Of_Artifacts.size() > 0 || this.List_Of_Creatures.size() > 0) && this.List_Of_Enchantments.size() > 0) {
			The_String_Builder.append("; ");
		}
		for (int i = 0; i < this.List_Of_Enchantments.size() - 1; i++) {
			The_String_Builder.append(this.List_Of_Enchantments.get(i));
			The_String_Builder.append("; ");
		}
		if (this.List_Of_Enchantments.size() > 0) {
			The_String_Builder.append(this.List_Of_Enchantments.get(this.List_Of_Enchantments.size() - 1));
		}
		
		if ((this.List_Of_Artifacts.size() > 0 || this.List_Of_Creatures.size() > 0 || this.List_Of_Enchantments.size() > 0) && this.List_Of_Lands.size() > 0) {
			The_String_Builder.append("; ");
		}
		for (int i = 0; i < this.List_Of_Lands.size() - 1; i++) {
			The_String_Builder.append(this.List_Of_Lands.get(i));
			The_String_Builder.append("; ");
		}
		if (this.List_Of_Lands.size() > 0) {
			The_String_Builder.append(this.List_Of_Lands.get(this.List_Of_Lands.size() - 1));
		}
		
		if ((this.List_Of_Artifacts.size() > 0 || this.List_Of_Creatures.size() > 0 || this.List_Of_Enchantments.size() > 0 || this.List_Of_Lands.size() > 0) && this.List_Of_Planeswalkers.size() > 0) {
			The_String_Builder.append("; ");
		}
		for (int i = 0; i < this.List_Of_Planeswalkers.size() - 1; i++) {
			The_String_Builder.append(this.List_Of_Planeswalkers.get(i));
			The_String_Builder.append("; ");
		}
		if (this.List_Of_Planeswalkers.size() > 0) {
			The_String_Builder.append(this.List_Of_Planeswalkers.get(this.List_Of_Planeswalkers.size() - 1));
		}

		return The_String_Builder.toString();
	}
	
}

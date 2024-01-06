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
	
	public ArrayList<an_artifact> list_of_artifacts() {
		return this.List_Of_Artifacts;
	}
	
	public ArrayList<a_creature> list_of_creatures() {
		return this.List_Of_Creatures;
	}
	
	public ArrayList<an_enchantment> list_of_enchantments() {
		return this.List_Of_Enchantments;
	}
	
	public ArrayList<a_land> list_of_lands() {
		return this.List_Of_Lands;
	}
	
	public ArrayList<a_permanent> list_of_permanents() {
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
	
	public ArrayList<a_planeswalker> list_of_planeswalkers() {
		return this.List_Of_Planeswalkers;
	}
	
	public int number_of_permanents() {
		return
			this.List_Of_Artifacts.size() +
			this.List_Of_Creatures.size() +
			this.List_Of_Enchantments.size() +
			this.List_Of_Lands.size() +
			this.List_Of_Planeswalkers.size();
	}
	
	public void receives(a_creature The_Creature) {
		this.List_Of_Creatures.add(The_Creature);
	}
	
	public void receives(a_land The_Land) {
		this.List_Of_Lands.add(The_Land);
	}
	
	@Override
	public String toString() {
		StringBuilder The_String_Builder = new StringBuilder("Part of the Battlefield: ");
		
		for (int i = 0; i < this.List_Of_Artifacts.size() - 1; i++) {
			an_artifact The_Artifact = this.List_Of_Artifacts.get(i);
			String The_String_Representing_The_Indicator_Of_Whether_The_Artifact_Is_Tapped = (The_Artifact.is_tapped()) ? "(T)" : "(U)";
			The_String_Builder.append(The_Artifact + " " + The_String_Representing_The_Indicator_Of_Whether_The_Artifact_Is_Tapped + "; ");
		}
		if (!this.List_Of_Artifacts.isEmpty()) {
			an_artifact The_Artifact = this.List_Of_Artifacts.get(this.List_Of_Artifacts.size() - 1);
			String The_String_Representing_The_Indicator_Of_Whether_The_Artifact_Is_Tapped = (The_Artifact.is_tapped()) ? "(T)" : "(U)";
			The_String_Builder.append(The_Artifact + " " + The_String_Representing_The_Indicator_Of_Whether_The_Artifact_Is_Tapped);
		}
		
		if (!this.List_Of_Artifacts.isEmpty() && !this.List_Of_Creatures.isEmpty()) {
			The_String_Builder.append("; ");
		}
		for (int i = 0; i < this.List_Of_Creatures.size() - 1; i++) {
			a_creature The_Creature = this.List_Of_Creatures.get(i);
			String The_String_Representing_The_Indicator_Of_Whether_The_Creature_Is_Tapped = (The_Creature.is_tapped()) ? "(T)" : "(U)";
			The_String_Builder.append(The_Creature + " " + The_String_Representing_The_Indicator_Of_Whether_The_Creature_Is_Tapped + "; ");
		}
		if (!this.List_Of_Creatures.isEmpty()) {
			a_creature The_Creature = this.List_Of_Creatures.get(this.List_Of_Creatures.size() - 1);
			String The_String_Representing_The_Indicator_Of_Whether_The_Creature_Is_Tapped = (The_Creature.is_tapped()) ? "(T)" : "(U)";
			The_String_Builder.append(The_Creature + " " + The_String_Representing_The_Indicator_Of_Whether_The_Creature_Is_Tapped);
		}
		
		if ((!this.List_Of_Artifacts.isEmpty() || !this.List_Of_Creatures.isEmpty()) && !this.List_Of_Enchantments.isEmpty()) {
			The_String_Builder.append("; ");
		}
		for (int i = 0; i < this.List_Of_Enchantments.size() - 1; i++) {
			an_enchantment The_Enchantment = this.List_Of_Enchantments.get(i);
			String The_String_Representing_The_Indicator_Of_Whether_The_Enchantment_Is_Tapped = (The_Enchantment.is_tapped()) ? "(T)" : "(U)";
			The_String_Builder.append(The_Enchantment + " " + The_String_Representing_The_Indicator_Of_Whether_The_Enchantment_Is_Tapped + "; ");
		}
		if (!this.List_Of_Enchantments.isEmpty()) {
			an_enchantment The_Enchantment = this.List_Of_Enchantments.get(this.List_Of_Enchantments.size() - 1);
			String The_String_Representing_The_Indicator_Of_Whether_The_Enchantment_Is_Tapped = (The_Enchantment.is_tapped()) ? "(T)" : "(U)";
			The_String_Builder.append(The_Enchantment + " " + The_String_Representing_The_Indicator_Of_Whether_The_Enchantment_Is_Tapped);
		}
		
		if ((!this.List_Of_Artifacts.isEmpty() || !this.List_Of_Creatures.isEmpty() || !this.List_Of_Enchantments.isEmpty()) && !this.List_Of_Lands.isEmpty()) {
			The_String_Builder.append("; ");
		}
		for (int i = 0; i < this.List_Of_Lands.size() - 1; i++) {
			a_land The_Land = this.List_Of_Lands.get(i);
			String The_String_Representing_The_Indicator_Of_Whether_The_Land_Is_Tapped = (The_Land.is_tapped()) ? "(T)" : "(U)";
			The_String_Builder.append(The_Land + " " + The_String_Representing_The_Indicator_Of_Whether_The_Land_Is_Tapped + "; ");
		}
		if (!this.List_Of_Lands.isEmpty()) {
			a_land The_Land = this.List_Of_Lands.get(this.List_Of_Lands.size() - 1);
			String The_String_Representing_The_Indicator_Of_Whether_The_Land_Is_Tapped = (The_Land.is_tapped()) ? "(T)" : "(U)";
			The_String_Builder.append(The_Land + " " + The_String_Representing_The_Indicator_Of_Whether_The_Land_Is_Tapped);
		}
		
		if ((!this.List_Of_Artifacts.isEmpty() || !this.List_Of_Creatures.isEmpty() || !this.List_Of_Enchantments.isEmpty() || !this.List_Of_Lands.isEmpty()) && !this.List_Of_Planeswalkers.isEmpty()) {
			The_String_Builder.append("; ");
		}
		for (int i = 0; i < this.List_Of_Artifacts.size() - 1; i++) {
			a_planeswalker The_Planeswalker = this.List_Of_Planeswalkers.get(i);
			String The_String_Representing_The_Indicator_Of_Whether_The_Planeswalker_Is_Tapped = (The_Planeswalker.is_tapped()) ? "(T)" : "(U)";
			The_String_Builder.append(The_Planeswalker + " " + The_String_Representing_The_Indicator_Of_Whether_The_Planeswalker_Is_Tapped + "; ");
		}
		if (this.List_Of_Planeswalkers.size() > 0) {
			a_planeswalker The_Planeswalker = this.List_Of_Planeswalkers.get(this.List_Of_Planeswalkers.size() - 1);
			String The_String_Representing_The_Indicator_Of_Whether_The_Planeswalker_Is_Tapped = (The_Planeswalker.is_tapped()) ? "(T)" : "(U)";
			The_String_Builder.append(The_Planeswalker + " " + The_String_Representing_The_Indicator_Of_Whether_The_Planeswalker_Is_Tapped);
		}

		return The_String_Builder.toString();
	}
}
package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;

/**
 * a_stack
 * 
 * Rule 405.2: The stack keeps track of the order that spells and/or abilities were added to [the stack]. Each time an object in put on the stack, the object is put on top of all objects already there.
 */
public class a_stack {

	private boolean Has_Resolved;
	private ArrayList<Object> List_Of_Spells_Nonmana_Activated_Abilities_And_Triggered_Abilities;
	private ArrayList<a_triggered_ability> List_Of_Triggered_Abilities_To_Be_Added_To_This;
	
	public a_stack() {
		this.List_Of_Spells_Nonmana_Activated_Abilities_And_Triggered_Abilities = new ArrayList<Object>();
		this.List_Of_Triggered_Abilities_To_Be_Added_To_This = new ArrayList<a_triggered_ability>();
	}
	
	public void adds_all_triggered_abilities_in_list_of_triggered_abilities_to_be_added_to_this_to_this() {
		this.List_Of_Spells_Nonmana_Activated_Abilities_And_Triggered_Abilities.addAll(this.List_Of_Triggered_Abilities_To_Be_Added_To_This);
		this.List_Of_Triggered_Abilities_To_Be_Added_To_This.clear();
	}
	
	public boolean contains_objects() {
		return !this.List_Of_Spells_Nonmana_Activated_Abilities_And_Triggered_Abilities.isEmpty();
	}
	
	public boolean has_resolved() {
		return this.Has_Resolved;
	}
	
	public boolean isEmpty() {
		return this.List_Of_Spells_Nonmana_Activated_Abilities_And_Triggered_Abilities.isEmpty();
	}
	
	public ArrayList<a_triggered_ability> list_of_triggered_abilities_to_be_added_to_this() {
		return this.List_Of_Triggered_Abilities_To_Be_Added_To_This;
	}
	
	public void receives(Object The_Object) throws Exception {
		if ((The_Object instanceof a_spell) || (The_Object instanceof a_nonmana_activated_ability) || (The_Object instanceof a_triggered_ability)) {
			this.List_Of_Spells_Nonmana_Activated_Abilities_And_Triggered_Abilities.add(The_Object);
		} else {
			throw new Exception("The stack only receives spells, nonmana activated abilties, and triggered abilities.");
		}
	}
	
	public void removes(Object The_Object_To_Remove) {
		this.List_Of_Spells_Nonmana_Activated_Abilities_And_Triggered_Abilities.remove(The_Object_To_Remove);
	}
	
	public void resolves_top_object() throws Exception {
		this.Has_Resolved = false;
		if (this.contains_objects()) {
			Object The_Object = this.top_object();
			System.out.println("The following top stack spell or ability is resolving. " + The_Object);
			if (The_Object instanceof a_spell) {
				a_spell The_Spell = (a_spell) The_Object;
				if (The_Spell instanceof a_permanent_spell) {
					a_permanent_spell The_Permanent_Spell = (a_permanent_spell) The_Spell;
					if (The_Permanent_Spell.has_a_target()) {
						if (The_Permanent_Spell.type().equals("Aura")) {
							// TODO
						} else if (The_Permanent_Spell.type().equals("Mutating Creature")) {
							// TODO
						}
					} else {
						String The_Type_Of_The_Permanent_Spell = The_Permanent_Spell.type();
						System.out.println(The_Permanent_Spell + " becomes a " + The_Type_Of_The_Permanent_Spell + " and enters the battlefield under " + The_Permanent_Spell.player() + "'s control.");
						if (The_Type_Of_The_Permanent_Spell.equals("Creature")) {
							a_nonland_card The_Nonland_Card = The_Permanent_Spell.nonland_card();
							a_creature_card The_Creature_Card = (a_creature_card) The_Nonland_Card;
							a_creature The_Creature = new a_creature(The_Permanent_Spell.name(), new ArrayList<a_static_ability>(), The_Creature_Card, The_Permanent_Spell.player());
							ArrayList<a_triggered_ability> The_List_Of_Triggered_Abilities = new ArrayList<a_triggered_ability>();
							for (String The_Line : The_Creature_Card.text()) {
								if (The_Line.startsWith("When ")) {
									int The_Position_Of_The_First_Pause = The_Line.indexOf(", ");
									a_triggered_ability The_Triggered_Ability = new a_triggered_ability(The_Line.substring(5, The_Position_Of_The_First_Pause), The_Line.substring(The_Position_Of_The_First_Pause + 2), The_Creature);
									The_List_Of_Triggered_Abilities.add(The_Triggered_Ability);
								}
							}
							The_Creature.sets_its_list_of_triggered_abilities_to(The_List_Of_Triggered_Abilities);
							String The_Name_Of_The_Creature = The_Creature.name();
							The_Permanent_Spell.player().part_of_the_battlefield().receives(The_Creature);
							System.out.println(The_Permanent_Spell.player() + "'s part of the battlefield contains the following permanents. " + The_Permanent_Spell.player().part_of_the_battlefield());
						    for (a_creature Another_Creature : The_Permanent_Spell.player().part_of_the_battlefield().list_of_creatures()) {
							    for (a_triggered_ability The_Triggered_Ability : Another_Creature.list_of_triggered_abilities()) {
							    	if (The_Triggered_Ability.event().equals(The_Name_Of_The_Creature + " enters the battlefield")) {
							    		this.List_Of_Triggered_Abilities_To_Be_Added_To_This.add(The_Triggered_Ability);
							    		System.out.println("The following triggered ability has been added to the list of triggered abilities to be added to the stack. " + The_Triggered_Ability);
							    	}
							    }	
						    }
						    for (a_creature Another_Creature : The_Permanent_Spell.player().other_player().part_of_the_battlefield().list_of_creatures()) {
							    for (a_triggered_ability The_Triggered_Ability : Another_Creature.list_of_triggered_abilities()) {
							    	if (The_Triggered_Ability.event().equals(The_Name_Of_The_Creature + " enters the battlefield")) {
							    		this.List_Of_Triggered_Abilities_To_Be_Added_To_This.add(The_Triggered_Ability);
							    		System.out.println("The following triggered ability has been added to the list of triggered abilities to be added to the stack. " + The_Triggered_Ability);
							    	}
							    }	
						    }
						}
					}
				}
				else if (The_Spell.type().equals("Instant") ) {
					// TODO
				} else if (The_Spell.type().equals("Sorcery")) {
					// TODO
				}
			}
			else if (The_Object instanceof an_ability) {
				if (The_Object instanceof a_nonmana_activated_ability) {
					// TODO
				} else if (The_Object instanceof a_triggered_ability) {
					a_triggered_ability The_Triggered_Ability = (a_triggered_ability) The_Object;
					if (The_Triggered_Ability.effect().contains("if")) {
						// TODO
					} else {
						if (The_Triggered_Ability.effect().contains("put a +1/+1 counter on each other creature you control named Charmed Stray.")) {
							boolean The_Triggered_Ability_Had_An_Effect = false;
							for (a_creature The_Creature : The_Triggered_Ability.permanent().player().part_of_the_battlefield().list_of_creatures()) {
								if (!The_Creature.equals(The_Triggered_Ability.permanent()) && The_Creature.name().equals("Charmed Stray")) {
									The_Creature.receives_a_plus_one_plus_one_counter();
									The_Triggered_Ability_Had_An_Effect = true;
								}
							}
							String verb = (The_Triggered_Ability_Had_An_Effect) ? "had" : "did not have";
							System.out.println("The following triggered ability " + verb + " an effect. " + The_Triggered_Ability);
						}
					}
				}
			}
			this.removes(The_Object);
			this.Has_Resolved = true;
		}
	}
	
	public Object top_object() {
		return this.List_Of_Spells_Nonmana_Activated_Abilities_And_Triggered_Abilities.get(this.List_Of_Spells_Nonmana_Activated_Abilities_And_Triggered_Abilities.size() - 1);
	}
	
	@Override
	public String toString() {
		StringBuilder The_String_Builder = new StringBuilder("Stack: ");
		for (int i = 0; i < this.List_Of_Spells_Nonmana_Activated_Abilities_And_Triggered_Abilities.size() - 1; i++) {
			The_String_Builder.append(this.List_Of_Spells_Nonmana_Activated_Abilities_And_Triggered_Abilities.get(i) + "; ");
		}
		if (!this.List_Of_Spells_Nonmana_Activated_Abilities_And_Triggered_Abilities.isEmpty()) {
			The_String_Builder.append(this.List_Of_Spells_Nonmana_Activated_Abilities_And_Triggered_Abilities.get(this.List_Of_Spells_Nonmana_Activated_Abilities_And_Triggered_Abilities.size() - 1));
		}
		return The_String_Builder.toString();
	}
}
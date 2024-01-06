package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;
import java.util.Collections;

import org.apache.commons.math3.random.RandomDataGenerator;

public class a_player {
	
	/** Rule 103.4: Each player draws a number of cards equal to their starting hand size, which is normally seven. */
	private static int STARTING_HAND_SIZE = 7;
	
	private a_deck Deck;
	private an_exile Exile;
	private a_graveyard Graveyard;
	private a_hand Hand;
	private boolean Has_Cast_A_Spell_Or_Activated_A_Nonmana_Activated_Ability;
	private boolean Has_Played_A_Land_This_Turn;
	private boolean Has_Priority;
	private int Index_Of_The_Present_Turn;
	private int Life;
	private ArrayList<a_creature> List_of_Attackers;
	private ArrayList<a_battle> List_Of_Battles;
	private ArrayList<a_creature> List_Of_Blockers;
	private ArrayList<a_permanent> List_Of_Permanents_That_Should_Be_Untapped;
	private a_mana_pool Mana_Pool;
	private String Name;
	private a_part_of_the_battlefield Part_Of_The_Battlefield;
	private String Step;
	private a_player Other_Player;
	private RandomDataGenerator Random_Data_Generator;
	private a_stack Stack;
	private boolean Was_Starting_Player;
	
	/**
	 * a_player
	 * 
	 * Rule 103.3: Each player begins the game with a starting life total of 20.
	 */
	public a_player(a_deck The_Deck_To_Use, String The_Name_To_Use, a_stack The_Stack_To_Use)
	{
		this.Deck = The_Deck_To_Use;
		this.Exile = new an_exile();
		this.Graveyard = new a_graveyard();
		this.Hand = new a_hand();
		this.Life = 20;
		this.List_of_Attackers = new ArrayList<a_creature>();
		this.List_Of_Battles = new ArrayList<a_battle>();
		this.List_Of_Blockers = new ArrayList<a_creature>();
		this.List_Of_Permanents_That_Should_Be_Untapped = new ArrayList<a_permanent>();
		this.Mana_Pool = new a_mana_pool(0, 0, 0, 0, 0, 0);
		this.Name = The_Name_To_Use;
		this.Part_Of_The_Battlefield = new a_part_of_the_battlefield();
		this.Random_Data_Generator = new RandomDataGenerator();
		this.Stack = The_Stack_To_Use;
	}

	public a_mana_pool acquires_mana_for(Object The_Object) throws Exception {
		a_nonland_card The_Nonland_Card = null;
		a_nonmana_activated_ability The_Nonmana_Activated_Ability = null;
		if (The_Object instanceof a_nonland_card) {
			The_Nonland_Card = (a_nonland_card) The_Object;
		} else if (The_Object instanceof a_nonmana_activated_ability) {
			The_Nonmana_Activated_Ability = (a_nonmana_activated_ability) The_Object;
		} else {
			throw new Exception("Mana is acquired only for a nonland card or a nonmana activated ability.");
		}
		a_mana_pool The_Mana_Pool = new a_mana_pool(0, 0, 0, 0, 0, 0);
		ArrayList<ArrayList<a_mana_ability>> The_List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_The_Object;
		if (The_Nonland_Card != null) {
			The_List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_The_Object = The_Nonland_Card.list_of_combinations_of_available_mana_abilities_sufficient_to_play_this();
		} else if (The_Nonmana_Activated_Ability != null) {
			The_List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_The_Object = The_Nonmana_Activated_Ability.list_of_combinations_of_available_mana_abilities_sufficient_to_play_this();			
		} else {
			throw new Exception("A nonland card and a nonmana activated ability were null.");
		}
		ArrayList<a_mana_ability> The_Combination_Of_Available_Mana_Abilities_Sufficient_To_Play_The_Object = this.chooses_a_combination_of_available_mana_abilities_sufficient_to_play_an_object_from(The_List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_The_Object);
		for (a_mana_ability The_Available_Mana_Ability : The_Combination_Of_Available_Mana_Abilities_Sufficient_To_Play_The_Object) {
			The_Mana_Pool.increases_by(The_Available_Mana_Ability.activates_and_contributes_a_mana_pool());
		}
		return The_Mana_Pool;
	}
	
	public void assigns_a_list_of_sufficient_combinations_of_available_mana_abilities_to_nonland_card_or_nonmana_activated_ability(Object The_Object) throws Exception {
		a_nonland_card The_Nonland_Card = null;
		a_nonmana_activated_ability The_Nonmana_Activated_Ability = null;
		if (The_Object instanceof a_nonland_card) {
			The_Nonland_Card = (a_nonland_card) The_Object;
		} else if (The_Object instanceof a_nonmana_activated_ability) {
			The_Nonmana_Activated_Ability = (a_nonmana_activated_ability) The_Object;
		} else {
			throw new Exception("assigns_a_list_of_sufficient_combinations_of_available_mana_to takes a nonland card or a nonmana activated ability.");
		}
		ArrayList<a_permanent> The_List_Of_Permanents = this.Part_Of_The_Battlefield.list_of_permanents();
		ArrayList<a_mana_ability> The_List_Of_Available_Mana_Abilities_For_The_Player = new ArrayList<>();
		for (a_permanent The_Permanent : The_List_Of_Permanents) {
			ArrayList<a_mana_ability> The_List_Of_Available_Mana_Abilities_For_The_Permanent = The_Permanent.provides_a_list_of_available_mana_abilities();
			The_List_Of_Available_Mana_Abilities_For_The_Player.addAll(The_List_Of_Available_Mana_Abilities_For_The_Permanent);
		}
		ArrayList<ArrayList<a_mana_ability>> The_List_Of_Combinations_Of_Available_Mana_Abilities = this.generates_a_list_of_combinations_of_elements_in(The_List_Of_Available_Mana_Abilities_For_The_Player);
		ArrayList<ArrayList<a_mana_ability>> The_List_Of_Sufficient_Combinations_Of_Available_Mana_Abilities = new ArrayList<>();
		for (ArrayList<a_mana_ability> The_Combination_Of_Available_Mana_Abilities : The_List_Of_Combinations_Of_Available_Mana_Abilities) {
			a_mana_pool The_Mana_Pool = new a_mana_pool(0, 0, 0, 0, 0, 0);
			for (a_mana_ability The_Mana_Ability : The_Combination_Of_Available_Mana_Abilities) {
				The_Mana_Pool.increases_by(The_Mana_Ability.indicates_the_mana_pool_it_would_contribute());
			}
			a_mana_cost The_Mana_Cost = null;
			if (The_Nonland_Card != null) {
				The_Mana_Cost = The_Nonland_Card.mana_cost();
			} else if (The_Nonmana_Activated_Ability != null) {
				The_Mana_Cost = The_Nonmana_Activated_Ability.mana_cost();
			} else {
				throw new Exception("A nonland card and a nonmana activated ability were null.");
			}
			if (The_Mana_Cost != null) {
				if (The_Mana_Pool.is_sufficient_for(The_Mana_Cost)) {
					The_List_Of_Sufficient_Combinations_Of_Available_Mana_Abilities.add(The_Combination_Of_Available_Mana_Abilities);
				}
			} else {
				throw new Exception("A mana cost was null");
			}
		}
		if (The_Nonland_Card != null) {
			The_Nonland_Card.receives(The_List_Of_Sufficient_Combinations_Of_Available_Mana_Abilities);
		} else if (The_Nonmana_Activated_Ability != null) {
			The_Nonmana_Activated_Ability.receives(The_List_Of_Sufficient_Combinations_Of_Available_Mana_Abilities);
		} else {
			throw new Exception("A nonland card and a nonmana activated ability were null.");
		}
	}
	
	public void assigns_a_list_of_sufficient_combinations_of_available_mana_abilities_to_nonland_hand_cards_in(ArrayList<a_nonland_card> The_List_Of_Nonland_Hand_Cards) throws Exception {
		for (a_nonland_card The_Nonland_Card : The_List_Of_Nonland_Hand_Cards) {
			this.assigns_a_list_of_sufficient_combinations_of_available_mana_abilities_to_nonland_card_or_nonmana_activated_ability(The_Nonland_Card);
		}
	}
	
	public void assigns_a_list_of_sufficient_combinations_of_available_mana_abilities_to_her_permanents_nonmana_activated_abilities() throws Exception {
		for (a_permanent The_Permanent : this.Part_Of_The_Battlefield.list_of_permanents()) {
			for (a_nonmana_activated_ability The_Nonmana_Activated_Ability : The_Permanent.provides_its_list_of_nonmana_activated_abilities()) {
				this.assigns_a_list_of_sufficient_combinations_of_available_mana_abilities_to_nonland_card_or_nonmana_activated_ability(The_Nonmana_Activated_Ability);
			}
		}
	}
	
	public void becomes_the_starting_player() {
		this.Was_Starting_Player = true;
	}
	
	
	public Object chooses_a_playable_nonland_card_or_an_activatable_nonmana_activated_ability_from(ArrayList<Object> The_List_Of_Playable_Nonland_Cards_And_Activatable_Nonmana_Activated_Abilities) {
		int The_Random_Index_Of_A_Playable_Nonland_Card_Or_An_Activatable_Nonmana_Activated_Ability_In_The_List = this.Random_Data_Generator.nextInt(0, The_List_Of_Playable_Nonland_Cards_And_Activatable_Nonmana_Activated_Abilities.size() - 1);
		Object The_Random_Playable_Nonland_Card_Or_Activatable_Nonmana_Activated_Ability = The_List_Of_Playable_Nonland_Cards_And_Activatable_Nonmana_Activated_Abilities.remove(The_Random_Index_Of_A_Playable_Nonland_Card_Or_An_Activatable_Nonmana_Activated_Ability_In_The_List);
		return The_Random_Playable_Nonland_Card_Or_Activatable_Nonmana_Activated_Ability;
	}
	
	
	public ArrayList<a_mana_ability> chooses_a_combination_of_available_mana_abilities_sufficient_to_play_an_object_from(ArrayList<ArrayList<a_mana_ability>> The_List_Of_Sufficient_Combinations_Of_Available_Mana_Abilities) {
		int The_Index_Of_The_Sufficient_Combination = this.Random_Data_Generator.nextInt(0, The_List_Of_Sufficient_Combinations_Of_Available_Mana_Abilities.size() - 1);
		ArrayList<a_mana_ability> The_Sufficient_Combination = The_List_Of_Sufficient_Combinations_Of_Available_Mana_Abilities.get(The_Index_Of_The_Sufficient_Combination);
		return The_Sufficient_Combination;
	}
	
	
	public void completes_her_beginning_phase() {
		
		System.out.println(this.Name + " is completing their beginning phase.");
		
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 501.1: The beginning phase consists of three steps, in this order: untap, upkeep, and draw.
		this.completes_her_untap_step();
		
		this.completes_her_upkeep_step();
		
		// Rule 103.7a: In a two-player game, the player who plays first skips the draw step (see rule 504, "Draw Step") of their first turn.
		if ((this.Was_Starting_Player) && (this.Index_Of_The_Present_Turn == 0)) {
			System.out.println("Because " + this.Name + " is the starting player and this is the first turn, the draw step is skipped.");
		}
		else {
			this.completes_her_draw_step();
		}

		// Rule 500.2: A phase or step in which players receive priority ends when the stack is empty and all players pass in succession.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire.
	}
	
	/**
	 * completes_her_beginning_of_combat_step
	 * 
	 * Rule 507.2: Second, the active player gets priority. (See [R]ule 117, "Timing and Priority.")
	 */
	public void completes_her_beginning_of_combat_step() {
		this.Has_Priority = true;
	}
	
	/**
	 * completes_her_declare_attackers_step
	 * 
	 * Rule 508.1: First, the active player declares attackers. This turn-based action doesn't use the stack.
	 * To declare attackers, the active player follows the steps below, in order.
	 * If at any point during the declaration of attackers, the active player is unable to comply with any of steps listed below, the declaration is illegal; the game returns to the moment before the declaration (see [R]ule 728, "Handling Illegal Actions").
	 * Rule 508.1a: The active player chooses which creatures that they control, if any, will attack.
	 * The creatures must be untapped, they can't also be battles, and each one must either have haste or have been controlled by the active player continuously since the turn began.
	 * Rule 508.1b: If the defending player controls any planeswalkers, is the protector of any battles, or the game allows the active player to attack multiple other players, the active player announces which player, planewalker, or battle each of the chosen creatures is attacking.
	 * Rule 508.1c: The active player checks each creature they control to see whether [the creature is] affected by any restrictions (effects that say a creature can't attack, or that [the creature] can't attack unless some condition is met).
	 * If any restrictions are being disobeyed, the declaration of attackers is illegal.
	 * Rule 508.1c: The active player checks each creature they control to see whether [the creature is] affected by any requirements (effects that say attacks if able, or that it attacks if some condition is met).
	 * If the number of requirements that are being obeyed is fewer than the maximum possible number of requirements that could be obeyed without disobeying any restrictions, the declaration of attackers is illegal.
	 * If a creature can't attack unless a player pays a cost, that player is not required to pay that cost, even if attacking with that creature would increase the number of requirements being obeyed.
	 * If a requirement that says a creature attacks if able during a certain turn refers to a turn with multiple combat phases, the creature attacks if able during each declare attackers step in that turn.
	 * Rule 508.1e: If any of the chosen creatures have banding or a "bands with other" ability, the active player announces which creatures, if any, are banded with which. (See [R]ule 702.22, "Banding.")
	 * Rule 508.1f: The active player taps the chosen creatures. Tapping a creature when [the creature is] declared as an attacker isn't a cost; attacking simply causes creatures to become tapped.
	 * @throws Exception 
	 */
	public void completes_her_declare_attackers_step() throws Exception {
		this.Has_Priority = true;
		for (a_creature The_Creature : this.Part_Of_The_Battlefield.list_of_creatures()) {
			if (!The_Creature.is_tapped() && !The_Creature.is_battle() && (The_Creature.has_haste() || The_Creature.has_been_controlled_by_the_active_player_continuously_since_the_turn_began()) && The_Creature.can_attack()) {
				if (The_Creature.must_attack() || (an_enumeration_of_states_of_a_coin.provides_a_state() == an_enumeration_of_states_of_a_coin.HEADS)) {
					The_Creature.sets_its_indicator_of_whether_it_is_attacking_to(true);
					Object The_Attackee = null;
					if (!this.Part_Of_The_Battlefield.list_of_planeswalkers().isEmpty() || !this.List_Of_Battles.isEmpty()) {
						ArrayList<Object> The_List_Of_Possible_Attackees = new ArrayList<>();
						The_List_Of_Possible_Attackees.add(this);
						for (a_planeswalker The_Planeswalker : this.Other_Player.Part_Of_The_Battlefield.list_of_planeswalkers()) {
							The_List_Of_Possible_Attackees.add(The_Planeswalker);
						}
						for (a_battle The_Battle : this.List_Of_Battles) {
							The_List_Of_Possible_Attackees.add(The_Battle);
						}
						int The_Index_Of_The_Attackee = this.Random_Data_Generator.nextInt(0, The_List_Of_Possible_Attackees.size() - 1);
						The_Attackee = The_List_Of_Possible_Attackees.get(The_Index_Of_The_Attackee);
					} else {
						The_Attackee = this.Other_Player;
					}
					if (The_Attackee != null) {
						The_Creature.attacks(The_Attackee);
						System.out.println("The creature " + The_Creature + " attacks " + The_Attackee + ".");
					} else {
						throw new Exception("The target is null");
					}
					The_Creature.taps();
					this.List_of_Attackers.add(The_Creature);
				}
			}
		}
		System.out.println("After " + this + " declares attackers, " + this + " reacts.");
		this.reacts();
		this.Has_Priority = false;
		System.out.println("After " + this + " reacts to " + this + " declaring attackers, " + this.Other_Player + " reacts.");
		this.Other_Player.reacts();
		this.Has_Priority = true;
	}
	
	public void declares_blockers() throws Exception {
		this.Has_Priority = true;
		ArrayList<a_creature> The_List_Of_Blockers = new ArrayList<>();
		for (a_creature The_Creature : this.Part_Of_The_Battlefield.list_of_creatures()) {
			if (!The_Creature.is_tapped() && !The_Creature.is_battle()) {
				if (The_Creature.must_block() || (an_enumeration_of_states_of_a_coin.provides_a_state() == an_enumeration_of_states_of_a_coin.HEADS)) {
					The_List_Of_Blockers.add(The_Creature);
				}
			}
		}
		for (a_creature The_Blocker : The_List_Of_Blockers) {
			ArrayList<a_creature> The_List_Of_Attackers_That_The_Blocker_Can_Block = new ArrayList<>();
			for (a_creature The_Attacker : this.List_of_Attackers) {
				if (The_Blocker.can_block(The_Attacker)) {
					The_List_Of_Attackers_That_The_Blocker_Can_Block.add(The_Blocker);
				}
			}
			int The_Index_Of_Attacker = this.Random_Data_Generator.nextInt(0, The_List_Of_Attackers_That_The_Blocker_Can_Block.size() - 1);
			a_creature The_Attacker = The_List_Of_Attackers_That_The_Blocker_Can_Block.get(The_Index_Of_Attacker);
			The_Blocker.blocks(The_Attacker);
			The_Attacker.becomes_blocked_by(The_Blocker);
		}
		this.Has_Priority = false;
	}
	
	public void chooses_damage_assignment_order_for_her_attackers() {
		for (a_creature The_Attacker : this.List_of_Attackers) {
			if (The_Attacker.is_blocked()) {
				Collections.shuffle(The_Attacker.list_of_blockers());
			}
		}
	}
	
	public void chooses_damage_assignment_order_for_her_blockers() {
		this.Has_Priority = true;
		for (a_creature The_Blocker : this.List_Of_Blockers) {
			Collections.shuffle(The_Blocker.list_of_blockees());
		}
		this.Has_Priority = false;
	}
	
	/**
	 * completes_her_declare_blockers_step
	 * 
	 * Rule 509.1: First, the defending player declares blockers. This turn-based action doesn't use the stack. To declare blockers, the defending player follows the steps below, in order.
	 * If at any point during the declaration of blockers, the defending player is unable to comply with any of the steps listed below, the declaration is illegal; the game returns to the moment before the declaration (see [R]ule 728, "Handling Illegal Actions").
	 * Rule 509.1a: The defending player chooses which creatures they control, if any, will block.
	 * The chosen creatures must be untapped and they can't also be battles.
	 * For each of the chosen creatures, the defending player chooses one creature for [the chosen creature] to block that is attacking [the defending] player, a planeswalker [the defending player] control[s], or a battle [the defending player] protect[s].
	 */
	public void completes_her_declare_blockers_step() throws Exception {
		this.Has_Priority = false;
		this.Other_Player.declares_blockers();
		this.Has_Priority = true;
		this.chooses_damage_assignment_order_for_her_attackers();
		this.Has_Priority = false;
		this.Other_Player.chooses_damage_assignment_order_for_her_blockers();
		this.Has_Priority = true;
		
	}
	
	public void has_her_attackers_assign_combat_damage() throws Exception {
		this.Has_Priority = true;
		for (a_creature The_Attacker : this.List_of_Attackers) {
			if (The_Attacker.is_blocked()) {
				ArrayList<a_creature> The_List_Of_Blockers = The_Attacker.list_of_blockers();
				int combat_damage_to_be_dealt = The_Attacker.effective_power();
				for (a_creature The_Blocker : The_List_Of_Blockers) {
					if (combat_damage_to_be_dealt <= The_Blocker.effective_toughness()) {
						The_Blocker.receives_combat_damage(combat_damage_to_be_dealt);
					} else {
						The_Blocker.receives_combat_damage(The_Blocker.effective_toughness());
						combat_damage_to_be_dealt -= The_Blocker.effective_toughness();
					}
				}
			} else {
				Object The_Attackee = The_Attacker.attackee();
				if (The_Attackee instanceof a_player) {
					a_player The_Player = (a_player) The_Attackee;
					The_Player.Life -= The_Attacker.effective_power();
				} else if (The_Attackee instanceof a_planeswalker) {
					a_planeswalker The_Planeswalker = (a_planeswalker) The_Attackee;
					The_Planeswalker.receives_combat_damage(The_Attacker.effective_power());
				} else if (The_Attackee instanceof a_battle) {
					throw new Exception("Not implemented");
				}
			}
		}
	}
	
	private void puts_cards_corresponding_to_creatures_dealt_lethal_damage_in_graveyard() {
		ArrayList<a_creature> The_List_Of_Creatures = new ArrayList<>();
		for (a_creature The_Creature : this.Part_Of_The_Battlefield.list_of_creatures()) {
			if (The_Creature.was_dealt_lethal_damage()) {
				The_List_Of_Creatures.add(The_Creature);
			}
		}
		this.Part_Of_The_Battlefield.list_of_creatures().removeAll(The_List_Of_Creatures);
		for (a_creature The_Creature : The_List_Of_Creatures) {
			this.Graveyard.receives(The_Creature.creature_card());	
		}
	}
	
	public void has_her_blockers_assign_combat_damage() {
		this.Has_Priority = true;
		for (a_creature The_Blocker : this.List_Of_Blockers) {
			ArrayList<a_creature> The_List_Of_Blockees = The_Blocker.list_of_blockees();
			int combat_damage_to_be_dealt = The_Blocker.effective_power();
			for (a_creature The_Blockee : The_List_Of_Blockees) {
				if (combat_damage_to_be_dealt <= The_Blockee.effective_toughness()) {
					The_Blockee.receives_combat_damage(combat_damage_to_be_dealt);
				} else {
					The_Blockee.receives_combat_damage(The_Blockee.effective_toughness());
					combat_damage_to_be_dealt -= The_Blockee.effective_toughness();
				}
			}
		}
		this.Has_Priority = false;
	}
	
	/**
	 * completes_her_combat_damage_step
	 */
	public void completes_her_combat_damage_step() throws Exception {
		this.has_her_attackers_assign_combat_damage();
		this.Has_Priority = false;
		this.Other_Player.has_her_blockers_assign_combat_damage();
		this.Has_Priority = true;
		this.puts_cards_corresponding_to_creatures_dealt_lethal_damage_in_graveyard();
		this.Has_Priority = false;
		this.Other_Player.puts_cards_corresponding_to_creatures_dealt_lethal_damage_in_graveyard();
		this.Has_Priority = true;
	}
	
	/**
	 * completes_her_end_of_combat_step
	 */
	public void completes_her_end_of_combat_step() {
		this.Has_Priority = true;
	}
	
	/**
	 * completes_her_combat_phase
	 * 
	 * Rule 506.2: During the combat phase, the active player is the attacking player; creatures that player controls may attack.
	 * During the combat phase of a two-player game, the nonactive player is the defending player; that player, planeswalkers they control, and battles they protect may be attacked.
	 * Rule 506.3a: Only a creature can attack or block. Only a player, a planeswalker, or a battle can be attacked.
	 * Rule 506.5: A creature attacks alone if it's the only creature declared as an attacker during the declare attackers step.
	 * A creature blocks alone if it's the only creature declared as a blocker during the declare blockers step.
	 * A creature is blocking alone if it's blocking but no other creatures are.
	 * @throws Exception 
 	 */
	public void completes_her_combat_phase() throws Exception {
		
		System.out.println(this.Name + " is completing their combat phase.");
		
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 506.1: The combat phase has five steps, which proceed in order: beginning of combat, declare attackers, declare blockers, combat damage, and end of combat.
		// The declare blockers and combat damage steps are skipped if no creatures are declared as attackers or put onto the battlefield attacking (see rule 508.8).
		// There are two combat damage steps if any attacking or blocking creature has first strike (see 702.7) or double strike (see 702.4).
		ArrayList<a_creature> List_Of_Attackers = new ArrayList<>();
		if (!List_Of_Attackers.isEmpty()) {
			this.completes_her_declare_blockers_step();
			this.completes_her_combat_damage_step();
		}
		
		int number_of_combat_damage_steps = 1;
		for (a_creature The_Creature : this.Part_Of_The_Battlefield.list_of_creatures()) {
			ArrayList<a_static_ability> The_List_Of_Static_Abilities = The_Creature.list_of_static_abilities();
			for (a_static_ability The_Static_Ability : The_List_Of_Static_Abilities) {
				if (The_Static_Ability.effect().equals("first strike") || The_Static_Ability.effect().equals("double strike")) {
					number_of_combat_damage_steps = 2;
				}
			}
		}
		for (int i = 0; i < number_of_combat_damage_steps; i++) {
			this.completes_her_combat_damage_step();
		}
		
		// Rule 500.2: A phase or step in which players receive priority ends when the stack is empty and all players pass in succession.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire... Effects that last "until end of combat" expire at the end of the combat phase.
	}
	
	
    public void completes_her_draw_step() {
		
		System.out.println("    " + this.Name + " is completing their draw step.");
		
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 504.1: First, the active player draws a card. This turn-based action doesn't use the stack.
		this.draws();
		
		// Rule 504.2: Second, the active player gets priority. (See rule 117, "Timing and Priority.")
		this.Has_Priority = true;
		
		// Rule 500.2: A phase or step in which players receive priority ends when the stack is empty and all players pass in succession.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire.
		
		this.Has_Priority = false;
	}
	
	
	public void completes_her_end_phase() {
		
		System.out.println(this.Name + " is completing their end phase.");
		
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 500.2: A phase or step in which players receive priority ends when the stack is empty and all players pass in succession.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire... Effects that last "until end of combat" expire at the end of the combat phase.
		
	}
	
	public void casts_a_spell_or_activates_a_nonmana_activated_ability(boolean Indicator_Of_Whether_Player_Is_Reacting) throws Exception {
		// Rule 601.2: To cast a spell is to [use a card to create a spell], put [the spell] on the stack, and pay its mana costs, so that [the spell] will eventually resolve and have its effect. Casting a spell includes proposal of the spell (rules 601.2a-d) and determination and payment of costs (rules 601.2f-h). To cast a spell, a player follows the steps listed below, in order. A player must be legally allowed to cast the spell to begin this process (see rule 601.3). If a player is unable to comply with the requirements of a step listed below while performing that step, the casting of the spell is illegal; the game returns to the moment before the casting of that spell was proposed (see rule 723, "Handling Illegal Actions").
		// Rule 601.2e: The game checks to see if the proposed spell can legally be cast. If the proposed spell is illegal, the game returns to the moment before the casting of that spell was proposed (see rule 723, "Handling Illegal Actions").
		
		System.out.println(this + " is considering casting a spell or activating a nonmana activated ability.");
		ArrayList<a_nonland_card> The_List_Of_All_Nonland_Hand_Cards = this.Hand.list_of_nonland_cards();
		ArrayList<a_nonland_card> The_List_Of_Nonland_Hand_Cards_Appropriate_For_The_Present_Step;
		if (Indicator_Of_Whether_Player_Is_Reacting) {
			The_List_Of_Nonland_Hand_Cards_Appropriate_For_The_Present_Step = new ArrayList<>();
			for (a_nonland_card The_Nonland_Hand_Card : The_List_Of_All_Nonland_Hand_Cards) {
				if (The_Nonland_Hand_Card.type().equals("Instant") || The_Nonland_Hand_Card.text().contains("Flash")) {
					The_List_Of_Nonland_Hand_Cards_Appropriate_For_The_Present_Step.add(The_Nonland_Hand_Card);
				}
			}
		} else {			
			The_List_Of_Nonland_Hand_Cards_Appropriate_For_The_Present_Step = The_List_Of_All_Nonland_Hand_Cards;
		}
		this.assigns_a_list_of_sufficient_combinations_of_available_mana_abilities_to_nonland_hand_cards_in(The_List_Of_Nonland_Hand_Cards_Appropriate_For_The_Present_Step);
		this.assigns_a_list_of_sufficient_combinations_of_available_mana_abilities_to_her_permanents_nonmana_activated_abilities();
		this.determines_whether_are_playable_the_cards_in(The_List_Of_Nonland_Hand_Cards_Appropriate_For_The_Present_Step);
		this.determines_whether_her_permanents_nonmana_activated_abilities_are_activatable();
		ArrayList<a_nonland_card> The_List_Of_Playable_Nonland_Hand_Cards = this.generates_a_list_of_playable_nonland_hand_cards_in(The_List_Of_Nonland_Hand_Cards_Appropriate_For_The_Present_Step);
		System.out.println(this.Name + " may cast a spell using a card in the following list. " + The_List_Of_Playable_Nonland_Hand_Cards);
		ArrayList<a_nonmana_activated_ability> The_List_Of_Activatable_Nonmana_Activated_Abilities = this.generates_a_list_of_activatable_nonmana_activated_abilities();
		System.out.println(this.Name + " may activate an ability in the following list. " + The_List_Of_Activatable_Nonmana_Activated_Abilities);
		ArrayList<Object> The_List_Of_Playable_Nonland_Hand_Cards_And_Activatable_Nonmana_Activated_Abilities = new ArrayList<>();
		The_List_Of_Playable_Nonland_Hand_Cards_And_Activatable_Nonmana_Activated_Abilities.addAll(The_List_Of_Playable_Nonland_Hand_Cards);
		The_List_Of_Playable_Nonland_Hand_Cards_And_Activatable_Nonmana_Activated_Abilities.addAll(The_List_Of_Activatable_Nonmana_Activated_Abilities);
		
		// Rule 601.2a: To propose the casting of a spell, a player first [uses a card to create a spell and puts the spell on] the stack. [The spell] becomes the topmost object on the stack. [The spell] has all the characteristics of the card... associated with it, and [the casting] player becomes its controller. The spell remains on the stack until it's countered, it resolves, or an effect moves it elsewhere.
		if (The_List_Of_Playable_Nonland_Hand_Cards_And_Activatable_Nonmana_Activated_Abilities.size() > 0) {
			Object The_Playable_Nonland_Hand_Card_Or_Activatable_Nonmana_Activated_Ability = this.chooses_a_playable_nonland_card_or_an_activatable_nonmana_activated_ability_from(The_List_Of_Playable_Nonland_Hand_Cards_And_Activatable_Nonmana_Activated_Abilities);
			System.out.println(this + " will play the nonland hand card or activate the nonmana activated ability " + The_Playable_Nonland_Hand_Card_Or_Activatable_Nonmana_Activated_Ability + ".");
			//a_mana_pool The_Mana_Pool_To_Use_To_Cast_A_Spell =
			this.acquires_mana_for(The_Playable_Nonland_Hand_Card_Or_Activatable_Nonmana_Activated_Ability);
			//this.Mana_Pool.increases_by(The_Mana_Pool_To_Use_To_Cast_A_Spell);
			//this.Mana_Pool.decreases_by(The_Mana_Pool_To_Use_To_Cast_A_Spell);
			// Rule 117.3c: If a player has priority when they cast a spell, activate an ability, or take a special action, that player receives priority afterward.
			if (The_Playable_Nonland_Hand_Card_Or_Activatable_Nonmana_Activated_Ability instanceof a_nonland_card) {
				a_nonland_card The_Playable_Nonland_Hand_Card = (a_nonland_card) The_Playable_Nonland_Hand_Card_Or_Activatable_Nonmana_Activated_Ability;
				this.Hand.removes(The_Playable_Nonland_Hand_Card);
				System.out.println("After playing a nonland card, the hand of " + this.Name + " has " + this.Hand.number_of_cards() + " cards and contains the following. " + this.Hand);
				String The_Type_Of_The_Playable_Nonland_Hand_Card = The_Playable_Nonland_Hand_Card.type();
				if (The_Type_Of_The_Playable_Nonland_Hand_Card.equals("Instant") || The_Type_Of_The_Playable_Nonland_Hand_Card.equals("Sorcery")) {
				    a_spell The_Spell = new a_spell(The_Playable_Nonland_Hand_Card.name(), The_Playable_Nonland_Hand_Card, this, The_Type_Of_The_Playable_Nonland_Hand_Card);
				    this.Stack.receives(The_Spell);
				    System.out.println(this.Name + " has cast instant or sorcery spell " + The_Spell + ".");
				} else {
					a_permanent_spell The_Permanent_Spell = new a_permanent_spell(The_Playable_Nonland_Hand_Card.name(), The_Playable_Nonland_Hand_Card, this, The_Type_Of_The_Playable_Nonland_Hand_Card);
					this.Stack.receives(The_Permanent_Spell);
				    System.out.println(this.Name + " has cast permanent spell " + The_Permanent_Spell + ".");
				}
				this.Has_Cast_A_Spell_Or_Activated_A_Nonmana_Activated_Ability = true;
				System.out.println("The stack contains the following spells and nonmana activated abilities. " + this.Stack);
				this.Has_Priority = false;
				System.out.println("After " + this.Name + " cast spell, " + this.Other_Player.Name + " reacts.");
				this.Other_Player.reacts();
				this.Has_Priority = true;
				System.out.println("After " + this.Name + " cast spell and " + this.Other_Player + " reacted, " + this + " continues a main phase.");
			} else if (The_Playable_Nonland_Hand_Card_Or_Activatable_Nonmana_Activated_Ability instanceof a_nonmana_activated_ability) {
				a_nonmana_activated_ability The_Nonmana_Activated_Ability = (a_nonmana_activated_ability) The_Playable_Nonland_Hand_Card_Or_Activatable_Nonmana_Activated_Ability;
				this.Stack.receives(The_Nonmana_Activated_Ability);
				System.out.println(this.Name + " has activated nonmana activated ability " + The_Nonmana_Activated_Ability + ".");
				this.Has_Cast_A_Spell_Or_Activated_A_Nonmana_Activated_Ability = true;
				System.out.println("The stack contains the following spells and nonmana activated abilities. " + this.Stack);
				this.Has_Priority = false;
				System.out.println("After " + this.Name + " activated nonmana activated ability, " + this.Other_Player.Name + " reacts.");
				this.Other_Player.reacts();
				this.Has_Priority = true;
				System.out.println("After " + this.Name + " cast spell and " + this.Other_Player + "reacted, " + this + " continues a main phase");
			}
		} else {
			this.Has_Cast_A_Spell_Or_Activated_A_Nonmana_Activated_Ability = false;
			System.out.println(this + " does not cast a spell and does not activate a nonmana activated ability.");
		}
	}
	
	public void completes_a_main_phase() throws Exception {
		System.out.println(this.Name + " is starting a main phase.");
		this.Step = "This Player's Precombat Main Phase";
		for (a_creature The_Creature : this.Part_Of_The_Battlefield.list_of_creatures()) {
			The_Creature.sets_its_indicator_of_whether_it_has_been_controlled_by_the_active_player_continuously_since_the_turn_began(true);
		}
		
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 505.4: Second, if the active player controls one or more Saga enchantments and it's the active player's precombat main phase, the active player puts a lore counter on each Saga they control. (See rule 714, "Saga Cards.") This turn-based action doesn't use the stack.

		// Rule 505.5: Third, the active player gets priority. (See rule 117, "Timing and Priority.")
		this.Has_Priority = true;
		
		// Rule 505.5b: During either main phase, the active player may play one land card from their hand if the stack is empty, if the player has priority, and if they haven't played a land this turn (unless an effect states the player may play additional lands). This action doesn't use the stack. Neither the land nor the action of playing the land is a spell or ability, so it can't be countered, and players can't respond to it with instants or activated abilities. (See rule 305, "Lands.")
		if (!this.Has_Played_A_Land_This_Turn) {
		    this.plays_a_land();
		}
		
		// Rule 505.5a: [A] main phase is the only phase in which a player can normally cast artifact, creature, enchantment, planeswalker, and sorcery spells. The active player may cast these spells.
		
		do {
			this.casts_a_spell_or_activates_a_nonmana_activated_ability(false);
		} while (this.Has_Cast_A_Spell_Or_Activated_A_Nonmana_Activated_Ability);
		
		// Rule 608.1: Each time all players pass in succession, the spell or ability on top of the stack resolves.
		// Rule 608.2: If the object that's resolving is an instant spell, a sorcery spell, or a[ nonmana activated ability or a triggered] ability, its resolution may involve several steps. The steps described in rules 608.2a and 608.2b are followed first. The steps described in rules 608.2c-k are then followed as appropriate, in no specific order. The steps described in rule 608.2m and 608.2n are followed last.
		// Rule 608.2b: If the spell or ability specifies targets, [the resolution] checks whether the targets are still legal.
		// Other changes to the game state may cause a target to no longer be legal; for example, its characteristics may have changed or an effect may have changed the text of the spell.
		// If the source of an ability has left the zone it was in, its last known information is used during this process.
		// If all its targets, for every instance of the word "target," are now illegal, the spell or ability doesn't resolve.
		// [The spell or ability is] removed from the stack and, if [the spell or ability is] a spell, [the corresponding nonland card is] put into its owner's graveyard.
		// Otherwise, the spell or ability will resolve normally.
		// Illegal targets, if any, won't be affected by parts of a resolving spell's effect for which [those targets] illegal.
		// Other parts of the effect for which those targets are not illegal may still affect [those targets].
		// If the spell or ability creates any continuous effects that affect game rules (see [R]ule 613.11), those effects to illegal targets.
		// If part of the effect requires information about an illegal target, [the effect] fails to determine any such information.
		// Any part of the effect that requires [such] information won't happen.
		// Rule 608.2c: The controller of the spell or ability follows [the spell or ability's] instructions in the order written.
		// However, replacement effects may modify these actions.
		// In some cases, later text on the [spell or ability] may modify the meaning of earlier text...
		// Don't just apply effects step by step without thinking in these cases--read the whole text and apply the rules of English to the text.
		// Rule 608.2d: If an effect of a spell or ability offers any choices other than choices made as part of casting the spell, activating the ability, or otherwise putting the spell or ability on the stack, the player announces these while applying the effect.
		// The player can't choose an option that's illegal or impossible, with the exception that having a library with no cards in [the library] doesn't make drawing a card an impossible action (see Rule 121.3).
		// If an effect divides or distributes something, such as damage or counters, as a player chooses among any number of untargeted players and/or objects, the player chooses the amount and division such that each chosen player or object receives at least one of whatever is being divided.
		// Rule 608.2e: Some spells or abilities have multiple steps or actions, denoted by separate sentences or clauses, that involve multiple players.
		// In these cases, the choices for the first action are made in APNAP order, and then the first action is processed simultaneously.
		// Then the choices for the second action are made in APNAP order, and then that action is processed simultaneously, and so on. See [R]ule 101.4.
		// Rule 608.2f: Some spells and abilities include actions taken on multiple players and/or objects.
		// In most cases, each such action is processed simultaneously.
		// If the action can't be processed simultaneously, it's instead processed considering each affected player or object individually.
		// APNAP order is used to make the primary determination of the order of those actions.
		// Secondarily, if the action is to be taken on both a player and an object they control or on multiple objects controlled by the same player, the player who controls the resolving spell or ability chooses the relative order of those actions.
		// Rule 608.2g: If an effect gives a player the option to pay mana, [the player] may activate mana abilities before taking that action.
		// If an effect specifically instructs or allows a player to cast a spell during resolution, [the player] does so by following the steps in rules 601.2a-i, except no player receives priority after [the spell is] cast.
		// That spell becomes the topmost object on the stack, and the currently resolving spell or ability continues to resolve, which may include casting other spells this way.
		// No other spells can normally be cast and no other abilities can normally be activated during resolution.
		// Rule 608.2h: If an effect requires information from the game (such as the number of creatures on the battlefield), the answer is determined only once, when the effect is applied.
		// If the effect requires information from a specific object, including the source of the ability itself, the effect uses the current information of that object if [the object is] in the public zone [the object] was expected to be in; if [the object is] no longer in that zone, or if the effect has moved [the object] from a public zone to a hidden zone, the effect uses the object's last known information.
		// See [R]ule 113.7a.
		// If an ability states that an object does something, [the object is] the object as it exists--or as it most recently existed--that [object] does [something], not the ability.
		// Rule 608.2i: If an effect refers to certain characteristics, it checks only for the value of the specified characteristics, regardless of any related ones an object may also have.
		// Rule 608.2k: If an instant spell, sorcery spell, or ability that can legally resolve leaves the stack once it starts to resolve, it will continue to resolve fully.
		while (this.Stack.contains_objects()) {
			this.Has_Priority = false;
			Object The_Object = this.Stack.top_object();
			System.out.println("The top stack spell or nonmana activated ability " + The_Object + " is resolving.");
			if (The_Object instanceof a_spell) {
				a_spell The_Spell = (a_spell) The_Object;
				// Rule 608.3: If the object that's resolving is a permanent spell, [the object's] resolution may involve several steps. The instructions in rules 608.3a and b are always performed first. Then one of the steps in rules 608.3c-e is performed, if appropriate.
				// Rule 608.3a: If the object that's resolving has no targets, it becomes a permanent and enters the battlefield under the control of the spell's controller.
				// Rule 608.3b: If the object that's resolving has a target, [the object's] resolution checks whether the target is still legal, as described in 608.2b. If a spell with an illegal target is a bestowed Aura spell (see [R]ule 702.103e) or a mutating creature spell (see [R]ule 702.140b), [the spell] becomes a creature spell and will resolve as described in [R]ule 608.3a. Otherwise, the spell doesn't resolve. [The spell] is removed from the stack and [the corresponding nonland card] is put into [the nonland card's] owner's graveyard.
				// Rule 608.3e: If a permanent spell resolves but its controller can't put [the resulting permanent] onto the battlefield, that player puts [the corresponding nonland card] into [the nonland card's] owner's graveyard.
				// Rule 608.3f: If the object that's resolving is a copy of a permanent spell, [the object] will become a token permanent as it is put onto the battlefield in any of the steps above. A token put onto the battlefield this way is no longer a copy of a spell and is not "created" for the purpose of any rules or effects that refer to creating a token.
				// Rule 608.g: If the object that's resolving has a static ability that functions on the stack and creates a delayed triggered ability, that delayed triggered ability is created as that permanent is put onto the battlefield in any of the steps above.
				if (The_Spell instanceof a_permanent_spell) {
					a_permanent_spell The_Permanent_Spell = (a_permanent_spell) The_Spell;
					if (The_Permanent_Spell.has_a_target()) {
						// Rule 608.3c: If the object that's resolving is an Aura spell, it becomes a permanent and is put onto the battlefield under the control of the spell's controller attached to the player or object [the spell] was targeting.
						if (The_Permanent_Spell.type().equals("Aura")) {
							// TODO
						}
						// Rule 608.3d: If the object that's resolving is a mutating creature spell, the object representing that spell merges with the permanent [the spell] is targeting (see [R]ule 725, "Merging with Permanents").
						else if (The_Permanent_Spell.type().equals("Mutating Creature")) {
							
						}
					} else {
						String The_Type_Of_The_Permanent_Spell = The_Permanent_Spell.type();
						System.out.println(The_Permanent_Spell.name() + " becomes a " + The_Type_Of_The_Permanent_Spell + " and enters the battlefield under the control of " + The_Permanent_Spell.player() + ".");
						if (The_Type_Of_The_Permanent_Spell.equals("Creature")) {
							a_nonland_card The_Nonland_Card = The_Permanent_Spell.nonland_card();
							a_creature_card The_Creature_Card = (a_creature_card) The_Nonland_Card;
							a_creature The_Creature = new a_creature(The_Permanent_Spell.name(), new ArrayList<a_static_ability>(), The_Creature_Card, this);
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
						    The_Permanent_Spell.player().Part_Of_The_Battlefield.receives(The_Creature);
						    System.out.println(The_Permanent_Spell.player() + "'s part of the battlefield contains the following permanents. " + The_Permanent_Spell.player().Part_Of_The_Battlefield);
						    for (a_creature Another_Creature : this.Part_Of_The_Battlefield.list_of_creatures()) {
							    for (a_triggered_ability The_Triggered_Ability : Another_Creature.list_of_triggered_abilities()) {
							    	if (The_Triggered_Ability.event().equals(The_Name_Of_The_Creature + " enters the battlefield")) {
							    		this.Stack.receives(The_Triggered_Ability);
							    		System.out.println(The_Triggered_Ability + " has been added to the stack.");
							    	}
							    }	
						    }
						    for (a_creature Another_Creature : this.Other_Player.Part_Of_The_Battlefield.list_of_creatures()) {
							    for (a_triggered_ability The_Triggered_Ability : Another_Creature.list_of_triggered_abilities()) {
							    	if (The_Triggered_Ability.event().equals(The_Name_Of_The_Creature + " enters the battlefield")) {
							    		this.Stack.receives(The_Triggered_Ability);
							    		System.out.println(The_Triggered_Ability + " has been added to the stack.");
							    	}
							    }	
						    }
						    System.out.println("After triggered abilities are added to the stack, " + this + " reacts.");
						    this.reacts();
						    System.out.println("After triggered abilities are added to the stack and " + this + " reacts, " + this.Other_Player + " reacts.");
						    this.Other_Player.reacts();
						}
					}
				}
				// Rule 608.2m: As the final part of an instant or sorcery spell's resolution, the [nonland card corresponding to the spell] is put into its owner's graveyard...
				else if (The_Spell.type().equals("Instant") ) {
					// TODO
				} else if (The_Spell.type().equals("Sorcery")) {
					// TODO
				}
			}
			// Rule 608.2j: If an ability's effect refers to a specific untargeted object that has been previously referred to by that ability's cost or trigger condition, it still affects that object even if the object has changed characteristics.
			// Rule 608.2m: ... As the final part of an ability's resolution, the ability is removed from the stack and ceases to exist.
			// Rule 608.2n: Once all possible steps described in 608.2c-m are completed, any abilities that trigger when that spell or ability resolves trigger.
			else if (The_Object instanceof an_ability) {
				if (The_Object instanceof a_nonmana_activated_ability) {
					// TODO
				} else if (The_Object instanceof a_triggered_ability) {
					a_triggered_ability The_Triggered_Ability = (a_triggered_ability) The_Object;
					// Rule 608.2a: If a triggered ability has an intervening "if" clause, [the resolution] checks whether the clause's condition is true. If [the condition] isn't, the ability is removed from the stack and does nothing. Otherwise, it continues to resolve.
					if (The_Triggered_Ability.effect().contains("if")) {
						// TODO
					} else {
						if (The_Triggered_Ability.effect().contains("put a +1/+1 counter on each other creature you control named Charmed Stray.")) {
							for (a_creature The_Creature : this.Part_Of_The_Battlefield.list_of_creatures()) {
								if (!The_Creature.equals(The_Triggered_Ability.permanent()) && The_Creature.name().equals("Charmed Stray")) {
									The_Creature.receives_a_plus_one_plus_one_counter();
								}
							}
						}
						System.out.println("Resolved " + The_Triggered_Ability.effect());
					}
				}
			}
			this.Stack.removes(The_Object);
			// Rule 117.3b: The active player receives priority after a spell or [nonmana activated ]ability (other than a mana ability) resolves.
			this.reacts();
			this.Other_Player.reacts();
		}
		
		// Rule 405.5: ... If the stack is empty when all players pass, the current step or phase ends and the next begins.
		// Rule 500.2: A phase or step in which players receive priority ends when the stack is empty and all players pass in succession.
		// Rule 505.2: The main phase has no steps, so a main phase ends when all players pass in succession while the stack is empty.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire... Effects that last "until end of combat" expire at the end of the combat phase.
		
		this.Has_Priority = false;
	}
	
	
	public void completes_her_untap_step() {
		
		System.out.println(this.Name + " is completing their untap step.");
		
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 502.1: First, all phased-in permanents with phasing that the active player controls phase out, and all phased-out permanents that the active player controlled when they phased out phase in. This all happens simultaneously. This turn-based action doesn't use the stack. See rule 702.25, "Phasing."

		// Rule 502.2: Second, the active player determines which permanents they control will untap. Then they untap them all simultaneously. This turn-based action doesn't use the stack. Normally, all of a player's permanents untap, but effects can keep one or more of a player's permanents from untapping.
		this.determines_her_permanents_to_untap();
		this.untaps_her_permanents();
		
		// Rule 502.3: No player receives priority during the untap step, so no spells can be cast or resolve and no abilities can be activated or resolve. Any ability that triggers during this step will be held until the next time a player would receive priority, which is usually during the upkeep step (See rule 503, "Upkeep Step.")
		// Rule 500.3: A step in which no players receive priority ends when all specified actions that take place during that step are completed.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire.
	}
	
	
	// TODO: Allow for adding abilities that triggered during the untap stap and abilities that triggered at the beginning of the upkeep step to the stack.
	// TODO: Allow for processing the stack.
	public void completes_her_upkeep_step() {
		
		System.out.println(this.Name + " is completing their upkeep step.");
		
		// Rule 500.5: When a phase or step begins, any effects scheduled to last "until" that phase or step expire.
		// Rule 500.6: When a phase or step begins, any abilities that trigger "at the beginning of" that phase or step trigger. They are put on the stack the next time a player would receive priority. (See rule 117, "Timing and Priority.")
		
		// Rule 503.1a: Any abilities that triggered during the untap step and any abilities that triggered at the beginning of the upkeep [step] are put onto the stack before the active player gets priority; the order in which they triggered doesn't matter. (See rule 603, "Handling Triggered Abilities.")
		
		// Rule 503.1: The upkeep step has no turn-based actions. Once it begins, the active player gets priority. (See rule 117, "Timing and Priority.")
		this.Has_Priority = true;
		
		// Rule 500.2: A phase or step in which players receive priority ends when the stack is empty and all players pass in succession.
		// Rule 500.4: When a step or phase ends, any unused mana left in a player's mana pool empties. This turn-based action doesn't use the stack.
		// Rule 500.5: When a phase or step ends, any effects scheduled to last "until end of" that phase or step expire.
		
		this.Has_Priority = false;
	}
	
	
	// TODO: Use discretion in determining permanents to untap.
	public void determines_her_permanents_to_untap() {
		
		System.out.println(this.Name + " is determining their permanents to untap.");
		
		this.List_Of_Permanents_That_Should_Be_Untapped.clear();
		
		for (an_artifact The_Artifact : this.Part_Of_The_Battlefield.list_of_artifacts()) {
			if (The_Artifact.is_tapped()) {
				this.List_Of_Permanents_That_Should_Be_Untapped.add(The_Artifact);
			}
		}
		for (a_creature The_Creature : this.Part_Of_The_Battlefield.list_of_creatures()) {
			if (The_Creature.is_tapped()) {
				this.List_Of_Permanents_That_Should_Be_Untapped.add(The_Creature);
			}
		}
		for (an_enchantment The_Enchantment : this.Part_Of_The_Battlefield.list_of_enchantments()) {
			if (The_Enchantment.is_tapped()) {
				this.List_Of_Permanents_That_Should_Be_Untapped.add(The_Enchantment);
			}
		}
		for (a_land The_Land : this.Part_Of_The_Battlefield.list_of_lands()) {
			if (The_Land.is_tapped()) {
				this.List_Of_Permanents_That_Should_Be_Untapped.add(The_Land);
			}
		}
		for (a_planeswalker The_Planeswalker : this.Part_Of_The_Battlefield.list_of_planeswalkers()) {
			if (The_Planeswalker.is_tapped()) {
				this.List_Of_Permanents_That_Should_Be_Untapped.add(The_Planeswalker);
			}
		}
	}
	
	
	public void determines_whether_are_playable_the_cards_in(ArrayList<a_nonland_card> The_List_Of_Nonland_Hand_Cards) {
		for (a_nonland_card The_Nonland_Card : The_List_Of_Nonland_Hand_Cards) {
			if (this.indicates_whether_is_playable(The_Nonland_Card)) {
				The_Nonland_Card.becomes_playable();
			} else {
				The_Nonland_Card.becomes_not_playable();
			}
		}
	}
	
	
	public void determines_whether_her_permanents_nonmana_activated_abilities_are_activatable() throws Exception {
		for (a_permanent The_Permanent : this.Part_Of_The_Battlefield.list_of_permanents()) {
			for (a_nonmana_activated_ability The_Nonmana_Activated_Ability : The_Permanent.provides_its_list_of_nonmana_activated_abilities()) {
				if (The_Nonmana_Activated_Ability.requires_tapping()) {
					if (The_Permanent.is_tapped()) {
						The_Nonmana_Activated_Ability.becomes_nonactivatable();
					} else {
						if (The_Nonmana_Activated_Ability.list_of_combinations_of_available_mana_abilities_sufficient_to_play_this().size() > 0) {
							The_Nonmana_Activated_Ability.becomes_activatable();
						} else {
							The_Nonmana_Activated_Ability.becomes_nonactivatable();
						}
					}
				} else {
					if (The_Nonmana_Activated_Ability.list_of_combinations_of_available_mana_abilities_sufficient_to_play_this().size() > 0) {
						The_Nonmana_Activated_Ability.becomes_activatable();
					} else {
						The_Nonmana_Activated_Ability.becomes_nonactivatable();
					}
				}
			}
		}
	}
	
	
	public void draws() {
		this.Hand.receives(this.Deck.provides_its_top_card());
		
		System.out.println(
			"After drawing, the deck of " + this.Name + " has " + this.Deck.number_of_cards() + " cards and contains the following. " + this.Deck
		);
		System.out.println(
			"After drawing, the hand of " + this.Name + " has " + this.Hand.number_of_cards() + " cards and contains the following. " + this.Hand
		);
	}
	
	
	/**
	 * draws_a_hand
	 * 
	 * Rule 103.4: Each player draws a number of cards equal to their starting hand size, which is normally seven.
	 */
	public void draws_a_hand() {
		for (int i = 0; i < STARTING_HAND_SIZE; i++) {
			this.draws();
		}
	}
    
	
    private <T> ArrayList<ArrayList<T>> generates_a_list_of_combinations_of_elements_in(ArrayList<T> list_of_elements) {
        ArrayList<ArrayList<T>> list_of_combinations = new ArrayList<>();
        this.helps_generate_a_list_of_combinations(list_of_elements, 0, new ArrayList<T>(), list_of_combinations);
        return list_of_combinations;
    }
    
    
    private ArrayList<a_nonland_card> generates_a_list_of_playable_nonland_hand_cards_in(ArrayList<a_nonland_card> The_List_Of_Nonland_Hand_Cards) {
		ArrayList<a_nonland_card> The_List_Of_Playable_Nonland_Hand_Cards = new ArrayList<a_nonland_card>();
		for (a_nonland_card The_Nonland_Card : The_List_Of_Nonland_Hand_Cards) {
			if (The_Nonland_Card.is_playable()) {
				The_List_Of_Playable_Nonland_Hand_Cards.add(The_Nonland_Card);
			}
		}
		return The_List_Of_Playable_Nonland_Hand_Cards;
    }
    
    
    private ArrayList<a_nonmana_activated_ability> generates_a_list_of_activatable_nonmana_activated_abilities() {
    	ArrayList<a_nonmana_activated_ability> The_List_Of_Activatable_Nonmana_Activated_Abilities = new ArrayList<a_nonmana_activated_ability>();
    	for (a_permanent The_Permanent : this.Part_Of_The_Battlefield.list_of_permanents()) {
    		for (a_nonmana_activated_ability The_Nonmana_Activated_Ability : The_Permanent.provides_its_list_of_nonmana_activated_abilities()) {
    			if (The_Nonmana_Activated_Ability.is_activatable()) {
    				The_List_Of_Activatable_Nonmana_Activated_Abilities.add(The_Nonmana_Activated_Ability);
    			}
    		}
    	}
    	return The_List_Of_Activatable_Nonmana_Activated_Abilities;
    }
    
    
    private <T> void helps_generate_a_list_of_combinations(ArrayList<T> list_of_elements, int starting_index_in_list_of_elements, ArrayList<T> combination, ArrayList<ArrayList<T>> list_of_combinations) {
        list_of_combinations.add(new ArrayList<>(combination));
        for (int i = starting_index_in_list_of_elements; i < list_of_elements.size(); i++) {
            combination.add(list_of_elements.get(i));
            helps_generate_a_list_of_combinations(list_of_elements, i + 1, combination, list_of_combinations);
            combination.remove(combination.size() - 1);
        }
    }
    
	
	public boolean indicates_whether_is_playable(a_nonland_card The_Nonland_Card) {
		if (
			this.indicates_whether_a_card_is_playable_according_to_the_text_of(The_Nonland_Card) &&
			!The_Nonland_Card.list_of_combinations_of_available_mana_abilities_sufficient_to_play_this().isEmpty()
		) {
			return true;
		} else {
			return false;
		}
	}
	
	
	public boolean indicates_whether_a_card_is_playable_according_to_the_text_of(a_card The_Card) {
		if (The_Card.type().equals("Instant")) {
			ArrayList<String> The_Text = The_Card.text();
			ArrayList<a_creature> The_List_Of_Creatures = this.Part_Of_The_Battlefield.list_of_creatures();
			
			for (String The_Line : The_Text) {
				if (The_Line.contains("creature you control") && The_List_Of_Creatures.isEmpty()) {
					return false;
				}
			}
			
			// Tactical Advantage: "Target blocking or blocked creature you control gets +2/+2 until end of turn."
			if (The_Text.contains("Target blocking or blocked creature you control")) {
				if (!this.Step.equals("This Player's Declare Blockers Step") && !this.Step.equals("Other Player's Declare Blockers Step")) {
					return false;
				} else {
					for (a_creature The_Creature : The_List_Of_Creatures) {
						if (The_Creature.is_blocked()) {
							return true;
						} else if (The_Creature.is_blocking()) {
							return true;
						}
					}
					return false;
				}
			}
			
			else {
				// Stony Strength: "Put a +1/+1 counter on target creature you control. Untap that creature."
				return true;
			}
			
		} else {
		    return true;
		}
	}
	
	
	public void plays_a_land() throws Exception {
		ArrayList<a_land_card> The_List_Of_Land_Cards = this.Hand.list_of_land_cards();
		int The_Number_Of_Land_Cards = The_List_Of_Land_Cards.size();
		if (The_Number_Of_Land_Cards > 0) {
			System.out.println(this.Name + " is playing a land.");
			int The_Index_Of_The_Land_Card_To_Play = this.Random_Data_Generator.nextInt(0, The_Number_Of_Land_Cards - 1);
			a_land_card The_Land_Card_To_Play = The_List_Of_Land_Cards.get(The_Index_Of_The_Land_Card_To_Play);
			String The_Name_Of_The_Land_Card = The_Land_Card_To_Play.name();
			a_land The_Land = null;
			if (The_Name_Of_The_Land_Card.equals("Plains")) {
				The_Land = new a_land("Plains", this);
				a_mana_ability The_Mana_Ability = new a_mana_ability("T", "Add [W].", The_Land);
				The_Land.receives(The_Mana_Ability);
			} else if (The_Name_Of_The_Land_Card.equals("Forest")) {
				The_Land = new a_land("Forest", this);
				a_mana_ability The_Mana_Ability = new a_mana_ability("T", "Add [G].", The_Land);
				The_Land.receives(The_Mana_Ability);
			} else {
				throw new Exception("The MTG Game Simulator does not know how to play the land with name " + The_Name_Of_The_Land_Card);
			}
			this.Hand.removes(The_Land_Card_To_Play);
			this.Part_Of_The_Battlefield.receives(The_Land);
			this.Has_Played_A_Land_This_Turn = true;
			System.out.println("After playing a land card, the hand of " + this.Name + " has " + this.Hand.number_of_cards() + " cards and contains the following. " + this.Hand);
			System.out.println("After playing a land card, the part of the battlefield of " + this.Name + " has " + this.Part_Of_The_Battlefield.number_of_permanents() + " cards and contains the following. " + this.Part_Of_The_Battlefield);
		}
	}
	
	
	public String name() {
		return this.Name;
	}
	
	
	/**
	 * reacts
	 * 
	 * When all players pass in succession, the top (last-added) spell or ability on the stack resolves...
	 */
	public void reacts() throws Exception {
		System.out.println(this.Name + " is reacting.");
		this.Has_Priority = true;
		this.Step = "Other Player's Precombat Main Phase";
		this.casts_a_spell_or_activates_a_nonmana_activated_ability(true);
		System.out.println(this.Name + " is passing.");
		this.Has_Priority = false;
		return;
	}
	
	
	public void receives(a_player The_Other_Player_To_Use) {
		this.Other_Player = The_Other_Player_To_Use;
	}
	
	
	public void shuffles_her_deck() {
		this.Deck.shuffles();
		System.out.println(
			"The deck of " + this.Name + " after shuffling has " + this.Deck.number_of_cards() + " cards and is the following. " + this.Deck
		);
	}
	
	
	public void takes_her_turn() throws Exception {

		System.out.println(this.Name + " is taking their turn.");
		
		this.completes_her_beginning_phase();
		this.completes_a_main_phase();
		this.completes_her_combat_phase();
		this.completes_a_main_phase();
		this.completes_her_end_phase();
		
		// Rule 500.5: Effects that last "until end of turn" are subject to special rules; see rule 514.2.	
	}
	
	
	public void untaps_her_permanents() {
		
		System.out.println(this.Name + " is untapping their permanents.");
		
		for (a_permanent The_Permanent : this.List_Of_Permanents_That_Should_Be_Untapped) {
			The_Permanent.untaps();
		}
	}
	
	
	@Override
	public String toString() {
		return this.Name;
	}

	
}
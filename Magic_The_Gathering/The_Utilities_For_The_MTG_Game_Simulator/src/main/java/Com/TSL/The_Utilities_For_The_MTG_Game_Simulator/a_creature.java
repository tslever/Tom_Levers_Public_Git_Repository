package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;

public class a_creature extends a_permanent {
	
	private boolean Indicator_Of_Whether_This_Creature_Has_Been_Controlled_By_The_Active_Player_Continuously_Since_The_Turn_Began = false;
	private boolean Indicator_Of_Whether_This_Creature_Is_Blocked = false;
	private boolean Indicator_Of_Whether_This_Creature_Is_Blocking = false;
	private boolean Indicator_Of_Whether_This_Creature_Will_Attack = false;
	private ArrayList<a_static_ability> List_Of_Static_Abilities;
	private ArrayList<a_creature> List_of_Blockees;
	private Object Attackee;
	private ArrayList<a_creature> List_Of_Blockers;
	
	public a_creature(String The_Name, ArrayList<a_static_ability> The_List_Of_Static_Abilities_To_Use) {
		super(The_Name);
		this.List_Of_Static_Abilities = The_List_Of_Static_Abilities_To_Use;
		this.List_Of_Blockers = new ArrayList<>();
	}
	
	public boolean can_attack() {
		return true;
	}
	
	public boolean can_block(a_creature The_Attacker) {
		return true;
	}
	
	public boolean has_been_controlled_by_the_active_player_continuously_since_the_turn_began() {
		return this.Indicator_Of_Whether_This_Creature_Has_Been_Controlled_By_The_Active_Player_Continuously_Since_The_Turn_Began;
	}
	
	public boolean has_haste() {
		for (a_static_ability The_Static_Ability : this.List_Of_Static_Abilities) {
			if (The_Static_Ability.effect().equals("haste")) {
				return true;
			}
		}
		return false;
	}
	
	public boolean is_battle() {
		return false;
	}
	
	public boolean is_blocked() {
		return this.Indicator_Of_Whether_This_Creature_Is_Blocked;
	}	
	
	public boolean is_blocking() {
		return this.Indicator_Of_Whether_This_Creature_Is_Blocking;
	}
	
	public ArrayList<a_static_ability> list_of_static_abilities() {
		return this.List_Of_Static_Abilities;
	}
	
	public boolean must_attack() {
		return false;
	}
	
	public boolean must_block() {
		return false;
	}
	
	public void sets_the_creatures_indicator_that_this_creature_has_been_controlled_by_the_active_player_continuously_since_the_turn_began(boolean The_Indicator_To_Use) {
		this.Indicator_Of_Whether_This_Creature_Has_Been_Controlled_By_The_Active_Player_Continuously_Since_The_Turn_Began = The_Indicator_To_Use;
	}
	
	public void sets_the_creatures_indicator_of_whether_this_creature_will_attack_to(boolean The_Indicator_To_Use) {
		this.Indicator_Of_Whether_This_Creature_Will_Attack = The_Indicator_To_Use;
	}
	
	public void attacks(Object The_Attackee) throws Exception {
		if ((The_Attackee instanceof a_player) || (The_Attackee instanceof a_planeswalker)) {
			this.Attackee = The_Attackee;
		} else {
			throw new Exception("The target is not a player or a planeswalker.");
		}
	}
	
	public void becomes_blocked_by(a_creature The_Blocker) {
		this.Indicator_Of_Whether_This_Creature_Is_Blocked = true;
		this.List_Of_Blockers.add(The_Blocker);
	}
	
	public void blocks(a_creature The_Blockee) throws Exception {
		this.Indicator_Of_Whether_This_Creature_Is_Blocking = true;
		this.List_of_Blockees.add(The_Blockee);
	}
	
	public ArrayList<a_creature> list_of_blockees() {
		return this.List_of_Blockees;
	}
	
	public ArrayList<a_creature> list_of_blockers() {
		return this.List_Of_Blockers;
	}
}
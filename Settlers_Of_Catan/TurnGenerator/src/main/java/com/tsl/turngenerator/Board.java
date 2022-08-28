package com.tsl.turngenerator;

public class Board {

	private final int NUMBER_OF_TILES = 19;
	private final int NUMBER_OF_COMMUNITIES = 54;
	
	private Tile[] tiles;
	private Community[] communities;
	private Road[] roads;
	
	public Board() {
		
		tiles = new Tile[NUMBER_OF_TILES];
		tiles[18] = new Tile( 4, -4, Resource.ORE,         new NumberToken( 5), false);
		tiles[17] = new Tile( 0, -6, Resource.BRICK,       new NumberToken( 2), false);
		tiles[16] = new Tile(-4, -8, Resource.LUMBER,      new NumberToken( 6), false);
		tiles[15] = new Tile(-6, -6, Resource.WOOL,        new NumberToken( 3), false);
		tiles[14] = new Tile(-8, -4, Resource.ORE,         new NumberToken( 8), false);
		tiles[13] = new Tile(-6,  0, Resource.GRAIN,       new NumberToken(10), false);
		tiles[12] = new Tile(-4,  4, Resource.GRAIN,       new NumberToken( 9), false);
		tiles[11] = new Tile( 0,  6, Resource.LUMBER,      new NumberToken(12), false);
		tiles[10] = new Tile( 4,  8, Resource.BRICK,       new NumberToken(11), false);
		tiles[ 9] = new Tile( 6,  6, Resource.GRAIN,       new NumberToken( 4), false);
		tiles[ 8] = new Tile( 8,  4, Resource.WOOL,        new NumberToken( 8), false);
		tiles[ 7] = new Tile( 6,  0, Resource.WOOL,        new NumberToken(10), false);
		tiles[ 6] = new Tile( 2, -2, Resource.LUMBER,      new NumberToken( 9), false);
		tiles[ 5] = new Tile(-2, -4, Resource.WOOL,        new NumberToken( 4), false);
		tiles[ 4] = new Tile(-4, -2, Resource.BRICK,       new NumberToken( 5), false);
		tiles[ 3] = new Tile(-2,  2, Resource.LUMBER,      new NumberToken( 6), false);
		tiles[ 2] = new Tile( 2,  4, Resource.ORE,         new NumberToken( 3), false);
		tiles[ 1] = new Tile( 4,  2, Resource.GRAIN,       new NumberToken(11), false);
		tiles[ 0] = new Tile( 0,  0, Resource.NO_RESOURCE, new NumberToken( 0),  true);
		
		communities = new Community[NUMBER_OF_COMMUNITIES];
		communities[53] = new Community(  6,  -4, Community.Type.NO_COMMUNITY, -1);
		communities[52] = new Community(  4,  -6, Community.Type.NO_COMMUNITY, -1);
		communities[51] = new Community(  2,  -6, Community.Type.NO_COMMUNITY, -1);
		communities[50] = new Community(  0,  -8, Community.Type.NO_COMMUNITY, -1);
		communities[49] = new Community( -2,  -8, Community.Type.NO_COMMUNITY, -1);
		communities[48] = new Community( -4, -10, Community.Type.NO_COMMUNITY, -1);
		communities[47] = new Community( -6, -10, Community.Type.NO_COMMUNITY, -1);
		communities[46] = new Community( -6,  -8, Community.Type.NO_COMMUNITY, -1);
		communities[45] = new Community( -8,  -8, Community.Type.NO_COMMUNITY, -1);
		communities[44] = new Community( -8,  -6, Community.Type.NO_COMMUNITY, -1);
		communities[43] = new Community(-10,  -6, Community.Type.NO_COMMUNITY, -1);
		communities[42] = new Community(-10,  -4, Community.Type.NO_COMMUNITY, -1);
		communities[41] = new Community( -8,  -2, Community.Type.NO_COMMUNITY, -1);
		communities[40] = new Community( -8,   0, Community.Type.NO_COMMUNITY, -1);
		communities[39] = new Community( -6,   2, Community.Type.NO_COMMUNITY, -1);
		communities[38] = new Community( -6,   4, Community.Type.NO_COMMUNITY, -1);
		communities[37] = new Community( -4,   6, Community.Type.NO_COMMUNITY, -1);
		communities[36] = new Community( -2,   6, Community.Type.NO_COMMUNITY, -1);
		communities[35] = new Community(  0,   8, Community.Type.NO_COMMUNITY, -1);
		communities[34] = new Community(  2,   8, Community.Type.NO_COMMUNITY, -1);
		communities[33] = new Community(  4,  10, Community.Type.NO_COMMUNITY, -1);
		communities[32] = new Community(  6,  10, Community.Type.NO_COMMUNITY, -1);
		communities[31] = new Community(  6,   8, Community.Type.NO_COMMUNITY, -1);
		communities[30] = new Community(  8,   8, Community.Type.NO_COMMUNITY, -1);
		communities[29] = new Community(  8,   6, Community.Type.NO_COMMUNITY, -1);
		communities[28] = new Community( 10,   6, Community.Type.NO_COMMUNITY, -1);
		communities[27] = new Community( 10,   4, Community.Type.NO_COMMUNITY, -1);
		communities[26] = new Community(  8,   2, Community.Type.NO_COMMUNITY, -1);
		communities[25] = new Community(  8,   0, Community.Type.NO_COMMUNITY, -1);
		communities[24] = new Community(  6,  -2, Community.Type.NO_COMMUNITY, -1);
		communities[23] = new Community(  4,  -2, Community.Type.NO_COMMUNITY, -1);
		communities[22] = new Community(  2,  -4, Community.Type.NO_COMMUNITY, -1);
		communities[21] = new Community(  0,  -4, Community.Type.NO_COMMUNITY, -1);
		communities[20] = new Community( -2,  -6, Community.Type.NO_COMMUNITY, -1);
		communities[19] = new Community( -4,  -6, Community.Type.NO_COMMUNITY, -1);
		communities[18] = new Community( -4,  -4, Community.Type.NO_COMMUNITY, -1);
		communities[17] = new Community( -6,  -4, Community.Type.NO_COMMUNITY, -1);
		communities[16] = new Community( -6,  -2, Community.Type.NO_COMMUNITY, -1);
		communities[15] = new Community( -4,   0, Community.Type.NO_COMMUNITY, -1);
		communities[14] = new Community( -4,   2, Community.Type.NO_COMMUNITY, -1);
		communities[13] = new Community( -2,   4, Community.Type.NO_COMMUNITY, -1);
		communities[12] = new Community(  0,   4, Community.Type.NO_COMMUNITY, -1);
		communities[11] = new Community(  2,   6, Community.Type.NO_COMMUNITY, -1);
		communities[10] = new Community(  4,   6, Community.Type.NO_COMMUNITY, -1);
		communities[ 9] = new Community(  4,   4, Community.Type.NO_COMMUNITY, -1);
		communities[ 8] = new Community(  6,   4, Community.Type.NO_COMMUNITY, -1);
		communities[ 7] = new Community(  6,   2, Community.Type.NO_COMMUNITY, -1);
		communities[ 6] = new Community(  4,   0, Community.Type.NO_COMMUNITY, -1);
		communities[ 5] = new Community(  2,   0, Community.Type.NO_COMMUNITY, -1);
		communities[ 4] = new Community(  0,  -2, Community.Type.NO_COMMUNITY, -1);
		communities[ 3] = new Community( -2,  -2, Community.Type.NO_COMMUNITY, -1);
		communities[ 2] = new Community( -2,   0, Community.Type.NO_COMMUNITY, -1);
		communities[ 1] = new Community(  0,   2, Community.Type.NO_COMMUNITY, -1);
		communities[ 0] = new Community(  2,   2, Community.Type.NO_COMMUNITY, -1);
		
		tiles[ 0].setCommunities(communities[ 0], communities[ 1], communities[ 2], communities[ 3], communities[ 4], communities[ 5]);
		tiles[ 1].setCommunities(communities[ 0], communities[ 5], communities[ 6], communities[ 7], communities[ 8], communities[ 9]);
		tiles[ 2].setCommunities(communities[ 0], communities[ 1], communities[ 9], communities[10], communities[11], communities[12]);
		// ...
		
		roads = new Road[72];
		
		// Community 0
		roads[ 0] = new Road(  5,  -5, Road.Type.NO_ROAD, -1);
		roads[ 1] = new Road(  6,  -3, Road.Type.NO_ROAD, -1);
		
		// Community 1
		roads[ 2] = new Road(  3,  -6, Road.Type.NO_ROAD, -1);
		
		// Community 2
		roads[ 3] = new Road(  1,  -7, Road.Type.NO_ROAD, -1);
		roads[ 4] = new Road(  2,  -5, Road.Type.NO_ROAD, -1);
		
		// Community 3
		roads[ 5] = new Road( -1,  -8, Road.Type.NO_ROAD, -1);
		
		// Community 4
		roads[ 6] = new Road( -3,  -9, Road.Type.NO_ROAD, -1);
		roads[ 7] = new Road( -2,  -7, Road.Type.NO_ROAD, -1);
		
		// Community 5
		roads[ 8] = new Road( -5, -10, Road.Type.NO_ROAD, -1);
		
		// Community 6
		roads[ 9] = new Road( -6,  -9, Road.Type.NO_ROAD, -1);		
		
		// Community 7
		roads[10] = new Road( -7,  -8, Road.Type.NO_ROAD, -1);
		roads[11] = new Road( -5,  -7, Road.Type.NO_ROAD, -1);
		
		// Community 8
		roads[12] = new Road( -8,  -7, Road.Type.NO_ROAD, -1);
		
		// Community 9
		roads[13] = new Road( -9,  -6, Road.Type.NO_ROAD, -1);
		roads[14] = new Road( -7,  -5, Road.Type.NO_ROAD, -1);
		
		// Community 10
		roads[15] = new Road(-10,  -5, Road.Type.NO_ROAD, -1);
		
		// Community 11
		roads[16] = new Road( -9,  -3, Road.Type.NO_ROAD, -1);
		
		// Community 12
		roads[17] = new Road( -8,  -1, Road.Type.NO_ROAD, -1);
		roads[18] = new Road( -7,  -2, Road.Type.NO_ROAD, -1);
		
		// Community 13
		roads[19] = new Road( -7,   1, Road.Type.NO_ROAD, -1);
		
		// Community 14
		roads[20] = new Road( -6,   3, Road.Type.NO_ROAD, -1);
		roads[21] = new Road( -5,   2, Road.Type.NO_ROAD, -1);
		
		// Community 15
		roads[22] = new Road( -5,   5, Road.Type.NO_ROAD, -1);
		
		// Community 16
		roads[23] = new Road( -3,   6, Road.Type.NO_ROAD, -1);
		
		// Community 17
		roads[24] = new Road( -1,   7, Road.Type.NO_ROAD, -1);
		roads[25] = new Road( -2,   5, Road.Type.NO_ROAD, -1);
		
		// Community 18
		roads[26] = new Road(  1,   8, Road.Type.NO_ROAD, -1);
		
		// Community 19
		roads[27] = new Road(  3,   9, Road.Type.NO_ROAD, -1);
		roads[28] = new Road(  2,   7, Road.Type.NO_ROAD, -1);
		
		// Community 20
		roads[29] = new Road(  5,  10, Road.Type.NO_ROAD, -1);
		
		// Community 21
		roads[30] = new Road(  6,   9, Road.Type.NO_ROAD, -1);
		
		// Community 22
		roads[31] = new Road(  7,   8, Road.Type.NO_ROAD, -1);
		roads[32] = new Road(  5,   7, Road.Type.NO_ROAD, -1);
		
		// Community 23
		roads[33] = new Road(  8,   7, Road.Type.NO_ROAD, -1);
		
		// Community 24
		roads[34] = new Road(  9,   6, Road.Type.NO_ROAD, -1);
		roads[35] = new Road(  7,   5, Road.Type.NO_ROAD, -1);
		
		// Community 25
		roads[36] = new Road( 10,   5, Road.Type.NO_ROAD, -1);
		
		// Community 26
		roads[37] = new Road(  9,   3, Road.Type.NO_ROAD, -1);
		
		// Community 27
		roads[38] = new Road(  8,   1, Road.Type.NO_ROAD, -1);
		roads[39] = new Road(  7,   2, Road.Type.NO_ROAD, -1);
		
		// Community 28
		roads[40] = new Road(  7,  -1, Road.Type.NO_ROAD, -1);
		
		// Community 29
		roads[41] = new Road(  5,  -2, Road.Type.NO_ROAD, -1);
		
		// Community 30
		roads[42] = new Road(  3,  -3, Road.Type.NO_ROAD, -1);
		roads[43] = new Road(  4,  -1, Road.Type.NO_ROAD, -1);
		
		// Community 31
		roads[44] = new Road(  1,  -4, Road.Type.NO_ROAD, -1);
		
		// Community 32
		roads[45] = new Road( -1,  -5, Road.Type.NO_ROAD, -1);
		roads[46] = new Road(  0,  -3, Road.Type.NO_ROAD, -1);
		
		// Community 33
		roads[47] = new Road( -3,  -6, Road.Type.NO_ROAD, -1);
		
		// Community 34
		roads[48] = new Road( -4,  -5, Road.Type.NO_ROAD, -1);
		
		// Community 35
		roads[49] = new Road( -5,  -4, Road.Type.NO_ROAD, -1);
		roads[50] = new Road( -3,  -3, Road.Type.NO_ROAD, -1);
		
		// Community 36
		roads[51] = new Road( -6,  -3, Road.Type.NO_ROAD, -1);
		
		// Community 37
		roads[52] = new Road( -5,  -1, Road.Type.NO_ROAD, -1);
		
		// Community 38
		roads[53] = new Road( -3,   0, Road.Type.NO_ROAD, -1);
		roads[54] = new Road( -4,   1, Road.Type.NO_ROAD, -1);
		
		// Community 39
		roads[55] = new Road( -3,   3, Road.Type.NO_ROAD, -1);
		
		// Community 40
		roads[56] = new Road( -1,   4, Road.Type.NO_ROAD, -1);
		
		// Community 41
		roads[57] = new Road(  0,   3, Road.Type.NO_ROAD, -1);
		roads[58] = new Road(  1,   5, Road.Type.NO_ROAD, -1);
		
		// Community 42
		roads[59] = new Road(  3,   6, Road.Type.NO_ROAD, -1);
		
		// Community 43
		roads[60] = new Road(  4,   5, Road.Type.NO_ROAD, -1);
		
		// Community 44
		roads[61] = new Road(  5,   4, Road.Type.NO_ROAD, -1);
		roads[62] = new Road(  3,   3, Road.Type.NO_ROAD, -1);
		
		// Community 45
		roads[63] = new Road(  6,   3, Road.Type.NO_ROAD, -1);
		
		// Community 46
		roads[64] = new Road(  5,   1, Road.Type.NO_ROAD, -1);
		
		// Community 47
		roads[65] = new Road(  3,   0, Road.Type.NO_ROAD, -1);
		
		// Community 48
		roads[66] = new Road(  1,  -1, Road.Type.NO_ROAD, -1);
		roads[67] = new Road(  2,   1, Road.Type.NO_ROAD, -1);
		
		// Community 49
		roads[68] = new Road( -1,  -2, Road.Type.NO_ROAD, -1);
		
		// Community 50
		roads[69] = new Road( -2,  -1, Road.Type.NO_ROAD, -1);
		
		// Community 51
		roads[70] = new Road( -1,   1, Road.Type.NO_ROAD, -1);
		roads[71] = new Road(  1,   2, Road.Type.NO_ROAD, -1);		
	}
	
	public Tile[] tiles() {
		return tiles;
	}
	
	public Community[] communities() {
		return communities;
	}
	
	public Road[] roads() {
		return roads;
	}
}

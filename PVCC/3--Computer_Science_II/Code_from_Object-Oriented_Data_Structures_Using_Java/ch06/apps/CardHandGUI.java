//---------------------------------------------------------------------
// CardHandGUI.java         by Dale/Joyce/Weems               Chapter 6
//
// Allows user to organize a hand of playing cards.
// Uses a graphical user innterface.
//----------------------------------------------------------------------
package ch06.apps;

import java.awt.*;
import javax.swing.*;
import java.awt.event.*;
import support.cards.*;

public class CardHandGUI 
{
  static final int HANDSIZE = 5;
  static int handCount = 0;
  static Card nextCard;
  static CardDeck deck = new CardDeck();
  static Card[] hand = new Card[HANDSIZE];
  static JLabel nextCardLabel;
  static JLabel nextCardHolder;
  static JLabel slotLabel;
  static JLabel[] cardLabels = new JLabel[HANDSIZE];
  static JLabel[] slotLabels = new JLabel[HANDSIZE];
  static ButtonGroup slotGroup;
  static JRadioButton[] rbuttons = new JRadioButton[HANDSIZE];

  public static void main (String[] args)
  {
    deck.shuffle();
    hand[0] = deck.nextCard();
    handCount = 1;
    nextCard = deck.nextCard();
   
    JFrame frame = new JFrame ("Card Hand GUI");
    frame.setDefaultCloseOperation (JFrame.EXIT_ON_CLOSE);

    JPanel mainPanel = new JPanel();
 //   mainPanel.setBackground (Color.blue);
    mainPanel.setPreferredSize (new Dimension (180 * HANDSIZE, 450));
    mainPanel.setLayout(new BorderLayout());

    // north panel
    JPanel northPanel = new JPanel();
    northPanel.setBackground (Color.white);
    northPanel.setLayout(new FlowLayout());

    nextCardLabel = new JLabel("Next Card:");
    nextCardHolder = new JLabel(nextCard.getImage());

    JPanel slotPanel = new JPanel();
    slotPanel.setLayout(new FlowLayout());
    slotLabel = new JLabel("Slot?");
    slotPanel.add(slotLabel);
 
    slotGroup = new ButtonGroup();
    SlotListener listener = new SlotListener();
    for (int i = 0; i < HANDSIZE; i++)
    {
      rbuttons[i] = new JRadioButton(Integer.toString(i));
      if (i > 1) rbuttons[i].setVisible(false);
      rbuttons[i].addActionListener(listener);
      slotGroup.add(rbuttons[i]);
      slotPanel.add(rbuttons[i]);
    }
    
    northPanel.add(nextCardLabel);
    northPanel.add(nextCardHolder);
    northPanel.add(slotPanel);

  
    // south panel
    for (int i = 0; i < HANDSIZE; i++)
      slotLabels[i] = new JLabel(Integer.toString(i));
      
    for (int i = 2; i < HANDSIZE; i++)
      slotLabels[i].setVisible(false);

    JPanel southPanel = new JPanel();
    southPanel.setBackground (Color.white);
    southPanel.setLayout(new FlowLayout());

    cardLabels[0] = new JLabel(hand[0].getImage());
    for (int i = 1; i < HANDSIZE; i++)
       cardLabels[i] = new JLabel("");
 
    for (int i = 0; i < HANDSIZE; i++)
    {
       southPanel.add(slotLabels[i]);
       southPanel.add(cardLabels[i]);
    }
    mainPanel.add(northPanel, BorderLayout.NORTH);
    mainPanel.add(southPanel, BorderLayout.SOUTH);
    
    frame.getContentPane().add(mainPanel);
    frame.pack();
    frame.setVisible(true);
  }
  
 private static class SlotListener implements ActionListener
 {
    public void actionPerformed (ActionEvent event)
    {
       Object source = event.getSource();
       handCount++;

       int target = 0;
       for (int i = 0; i < HANDSIZE; i++)
         if (source == rbuttons[i])
           target = i;

       for (int i = handCount - 1; i > target; i--)
         hand[i] = hand[i-1];
       hand[target] = nextCard;
       
       for (int i = 0; i < handCount; i++)
         cardLabels[i].setIcon(hand[i].getImage());

       if (handCount != HANDSIZE)
       {
         slotLabels[handCount].setVisible(true);
         rbuttons[handCount].setVisible(true);
         nextCard = deck.nextCard();
         nextCardHolder.setIcon(nextCard.getImage());
         slotGroup.clearSelection();
       }
       else
       {
         for (int i = 0; i < HANDSIZE; i++)
         {
           slotLabels[i].setVisible(false);
           rbuttons[i].setVisible(false);
         }
         slotLabel.setVisible(false);
         nextCardHolder.setVisible(false);
         nextCardLabel.setText("All Finished.");
       }
     }
  }
}
   


















































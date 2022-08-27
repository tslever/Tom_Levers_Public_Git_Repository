//----------------------------------------------------------------------
// SimGUI.java             by Dale/Joyce/Weems                 Chapter 5
//
// Simulates customers waiting in queues. Customers always enter
// the shortest queue.
//
// Input consists of simulation instance information:
//    Minimum and maximum customer interarrival time.
//    Minimum and maximum customer service time.
//    Number of queues and customers.
//
// Assumes a friendly user, i.e., inputs are valid.
//
// Output is the average waiting time for a customer.
// Uses a graphical user interface.
//----------------------------------------------------------------------
package ch04.apps;

import ch04.simulation.Simulation;
import java.awt.*;            
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import java.io.*;

public class SimulationGUI
{
  // input text fields
  private static JTextField minIATText;
  private static JTextField maxIATText;
  private static JTextField minSTText;
  private static JTextField maxSTText;
  private static JTextField numQueuesText;
  private static JTextField numCustText;

  // status Label
  private static JLabel resultLabel;         // label for status/result info

  // Define a button listener.
  private static class ActionHandler implements ActionListener 
  {
    public void actionPerformed(ActionEvent event)
    // listener for the button events
    {
      if (event.getActionCommand().equals("Simulate"))
      { // Handles Evaluate event.
        float result;
        Simulation sim = new Simulation
                  (Integer.parseInt(minIATText.getText()),
                   Integer.parseInt(maxIATText.getText()),
                   Integer.parseInt(minSTText.getText()),
                   Integer.parseInt(maxSTText.getText()));
 
        sim.simulate(Integer.parseInt(numQueuesText.getText()),
                     Integer.parseInt(numCustText.getText()));
        result = sim.getAvgWaitTime();
        
        resultLabel.setText("   Average Wait Time:  " + result);
      }
      else
      if (event.getActionCommand().equals("Clear"))
      { // Handles Clear event.
        resultLabel.setText("   RESULT  ");
        minIATText.setText("");
        maxIATText.setText("");
        minSTText.setText("");
        maxSTText.setText("");
        numQueuesText.setText("");
        numCustText.setText("");
   }
    }
  }

  public static void main(String args[]) throws IOException
  {
    // Declare/instantiate/initialize display frame.
    JFrame displayFrame = new JFrame();
    displayFrame.setTitle("Queue Simulation Program");
    displayFrame.setSize(350,300);
    displayFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

    // text boxes for input
    minIATText     = new JTextField("", 5);
    maxIATText     = new JTextField("", 5);
    minSTText      = new JTextField("", 5);
    maxSTText      = new JTextField("", 5);
    numQueuesText  = new JTextField("", 5);
    numCustText    = new JTextField("", 5);

    // input/result labels
    JLabel InArgLabel     = new JLabel("  Input Arguments:", JLabel.LEFT);
    JLabel minIATLabel    = new JLabel("    Minimum Interarrival Time", JLabel.CENTER);
    JLabel maxIATLabel    = new JLabel("    Maximum Interarrival Time", JLabel.CENTER);
    JLabel minSTLabel     = new JLabel("    Minimum Service Time", JLabel.CENTER);
    JLabel maxSTLabel     = new JLabel("    Maximum Service Time", JLabel.CENTER);
    JLabel numQueuesLabel = new JLabel("    Number of Queues", JLabel.CENTER);
    JLabel numCustLabel   = new JLabel("    Number of Customers", JLabel.CENTER);
    
    resultLabel    = new JLabel("   RESULT  ", JLabel.CENTER);

    // input/result panels
    JPanel minIATPanel    = new JPanel();
    JPanel maxIATPanel    = new JPanel();
    JPanel minSTPanel     = new JPanel();
    JPanel maxSTPanel     = new JPanel();
    JPanel numQueuesPanel = new JPanel();
    JPanel numCustPanel   = new JPanel();
    JPanel resultPanel    = new JPanel();
    
    // set input/result panel layouts
    minIATPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
    maxIATPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
    minSTPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
    maxSTPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
    numQueuesPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
    numCustPanel.setLayout(new FlowLayout(FlowLayout.LEFT));
    resultPanel.setLayout(new FlowLayout(FlowLayout.LEFT));

    // add items to input/result panels
    minIATPanel.add(minIATLabel);
    minIATPanel.add(minIATText);
    maxIATPanel.add(maxIATLabel);
    maxIATPanel.add(maxIATText);
    minSTPanel.add(minSTLabel);
    minSTPanel.add(minSTText); 
    maxSTPanel.add(maxSTLabel);
    maxSTPanel.add(maxSTText);
    numQueuesPanel.add(numQueuesLabel);
    numQueuesPanel.add(numQueuesText);
    numCustPanel.add(numCustLabel);
    numCustPanel.add(numCustText);
    
    resultPanel.add(resultLabel);
    
    // Simulate and Clear buttons    
    JButton simulate   = new JButton("Simulate");         
    JButton clear      = new JButton("Clear");         

    // Button event listener
    ActionHandler action = new ActionHandler();
 
    // Register button listeners.
    simulate.addActionListener(action);
    clear.addActionListener(action);

    // Instantiate content pane and information panels.
    Container contentPane = displayFrame.getContentPane();
    JPanel mainPanel = new JPanel();
    JPanel buttonPanel = new JPanel();
  
    // Initialize main panel.
    mainPanel.setLayout(new GridLayout(7,1));
    mainPanel.add(InArgLabel);
    mainPanel.add(minIATPanel);
    mainPanel.add(maxIATPanel);
    mainPanel.add(minSTPanel);
    mainPanel.add(maxSTPanel);
    mainPanel.add(numQueuesPanel);
    mainPanel.add(numCustPanel);

    // Initialize button panel.
    buttonPanel.setLayout(new GridLayout(1,2));
    buttonPanel.add(simulate);
    buttonPanel.add(clear);

    // Set up and show the frame.
    contentPane.add(mainPanel, "North");
    contentPane.add(buttonPanel, "Center");
    contentPane.add(resultPanel, "South");
 
    displayFrame.pack();
    displayFrame.setVisible(true);
  }
}

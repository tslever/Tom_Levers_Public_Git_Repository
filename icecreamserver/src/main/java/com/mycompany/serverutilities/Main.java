
// Allows executable to find class Main.
package com.mycompany.serverutilities;

// Imports classes.
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import org.apache.commons.lang.StringUtils;

/**
 * Defines class Main that encapsulates static method main, the entry point of
 * this application.
 * @version 0.0
 * @author Tom Lever
 */
public class Main {
    
    /**
     * Defines method main of this application, which instantiates a server
     * that listens for HTTP messages from clients (e.g., instances of Google
     * Chrome) and provides responses to clients.
     * @param args
     * @throws Exception
     * @throws IOException
     */
    public static void main(String[] args) throws Exception, IOException {
        System.out.println("main: Started Tom's ice-cream server program.");
        
        Config config = new Config();
        System.out.println(
            "main: Created a Config object to allow getting message " +
            "interfaces.");
        
        // isNumeric returns true if argument is a non-negative integer.
        if ((args.length != 1) || (!StringUtils.isNumeric(args[0]))) {
            throw new Exception(
                "main: First command-line argument has been found to not " +
                "exist or not be a non-negative integer.");
        }
        // TODO: Generate InvalidPortNumberException when args[0] is found to
        // not exist or to not be a non-negative integer.
        
        int portNumber = Integer.parseInt(args[0]);
        System.out.printf(
            "main: Parsed first command-line argument as port number %d.\n",
            portNumber);
        
        Server server = new Server(
            HttpServer.create(new InetSocketAddress(portNumber), 0));
        // TODO: Generate CreateException when IOException is thrown by create.
        System.out.println(
            "main: Created server listening on IP address : port " +
            "'localhost : " +
            portNumber +
            "' for HTTP messages.");
        
        server.setMessageInterfaces(config.getMessageInterfaces());
        // TODO: Generate SetMessageInterfacesException when an Exception is
        // thrown by setMessageInterfaces.
        System.out.println(
            "main: Set message interfaces of server as the array of message " +
            "interfaces resulting from our Config object's method " +
            "getMessageInterfaces.");
        
        server.startListeningForMessages();
        // TODO: Generate StartListeningForMessagesException when an Exception
        // is thrown by startListeningForMessages.
        System.out.println(
            "main: Started server listening for HTTP messages from clients.");
    }
}

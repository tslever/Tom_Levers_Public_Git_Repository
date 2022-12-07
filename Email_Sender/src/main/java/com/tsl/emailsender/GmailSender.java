package com.tsl.emailsender;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.security.GeneralSecurityException;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

import com.google.api.client.auth.oauth2.Credential;
import com.google.api.client.extensions.java6.auth.oauth2.AuthorizationCodeInstalledApp;
import com.google.api.client.extensions.jetty.auth.oauth2.LocalServerReceiver;
import com.google.api.client.googleapis.auth.oauth2.GoogleAuthorizationCodeFlow;
import com.google.api.client.googleapis.auth.oauth2.GoogleClientSecrets;
import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.googleapis.json.GoogleJsonError;
import com.google.api.client.googleapis.json.GoogleJsonResponseException;
import com.google.api.client.http.javanet.NetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.gson.GsonFactory;
import com.google.api.client.util.store.FileDataStoreFactory;
import com.google.api.services.gmail.Gmail;
import com.google.api.services.gmail.GmailScopes;
import com.google.api.services.gmail.model.Message;

import jakarta.activation.DataHandler;
import jakarta.activation.DataSource;
import jakarta.mail.MessagingException;
import jakarta.mail.Multipart;
import jakarta.mail.Session;
import jakarta.mail.internet.InternetAddress;
import jakarta.mail.internet.MimeBodyPart;
import jakarta.mail.internet.MimeMessage;
import jakarta.mail.internet.MimeMultipart;

/**
 * A {@code GmailSender} object represents a Gmail sender.
 */
public class GmailSender extends EmailSender {
    
    /**
     * Constructs a Gmail sender based on a {@code Configurations} object.
     * 
     * @param configurationsToUse a {@code Configurations} object with which to construct a Gmail sender
     * @param emailInfoToUse an {@code EmailInfo} object with which to construct a Gmail sender
     * @throws IOException if email information cannot be gotten
     */
    public GmailSender(Configurations configurationsToUse, EmailInfo emailInfoToUse) throws IOException {
        super(configurationsToUse, emailInfoToUse);
    }

    /**
     * Creates an authorized Credential object.
     * 
     * @param fromAddress the from address of an email
     * @param httpTransport a network HTTP Transport.
     * @param jsonFactory a JSON factory with which to load Google client secrets
     * @param pathToDirectoryContainingCredentials the path to a directory including credentials
     * @return an authorized Credential object.
     * @throws IOException if the oauth2ClientCredentials.json file cannot be found.
     */
    private static Credential getCredentials(String fromAddress, final NetHttpTransport httpTransport, JsonFactory jsonFactory, String pathToDirectoryContainingCredentials) throws IOException {
        // Download oauth2ClientCredentials.json from https://console.cloud.google.com/apis/credentials?project=<name-of-google-cloud-project>.
        FileReader fileReader = new FileReader(pathToDirectoryContainingCredentials + "/oauth2ClientCredentials.json");
        GoogleClientSecrets googleClientSecrets = GoogleClientSecrets.load(jsonFactory, fileReader);
        String scopeForSendingEmail = GmailScopes.GMAIL_SEND;
        List<String> listOfScopes = Collections.singletonList(scopeForSendingEmail);
        GoogleAuthorizationCodeFlow.Builder googleAuthorizationCodeFlowBuilder = new GoogleAuthorizationCodeFlow.Builder(httpTransport, jsonFactory, googleClientSecrets, listOfScopes);
        File directoryForAccessToken = new File(pathToDirectoryContainingCredentials);
        FileDataStoreFactory fileDataStoreFactory = new FileDataStoreFactory(directoryForAccessToken);
        googleAuthorizationCodeFlowBuilder.setDataStoreFactory(fileDataStoreFactory);
        googleAuthorizationCodeFlowBuilder.setAccessType("offline");
        GoogleAuthorizationCodeFlow googleAuthorizationCodeFlow = googleAuthorizationCodeFlowBuilder.build();
        LocalServerReceiver.Builder localServerReceiverBuilder = new LocalServerReceiver.Builder();
        localServerReceiverBuilder.setPort(8888);
        LocalServerReceiver localServerReceiver = localServerReceiverBuilder.build();
        AuthorizationCodeInstalledApp authorizationCodeInstalledApp = new AuthorizationCodeInstalledApp(googleAuthorizationCodeFlow, localServerReceiver);
        Credential credential = authorizationCodeInstalledApp.authorize("user");
        return credential;
    }
    
    /**
     * Sends an email using Gmail.
     * 
     * @throws IOException if an email header or body cannot be read,
     *                     a MIME message cannot be written to an output stream,
     *                     a new trusted transport cannot be constructed,
     *                     credentials cannot be gotten, or
     *                     an email cannot be sent
     * @throws MessagingException if a MIME message cannot be constructed or written to an output stream
     * @throws GeneralSecurityException if a new trusted transport cannot be constructed
     */
    @Override
    public void send() throws IOException, MessagingException, GeneralSecurityException {        
        // Encode as MIME message
        Properties properties = new Properties();
        Session session = Session.getDefaultInstance(properties, null);
        MimeMessage mimeMessage = new MimeMessage(session);
        mimeMessage.setSubject(emailInfo.getSubject());
        mimeMessage.setFrom(emailInfo.getFromAddress());
        List<String> arrayListOfToAddresses = emailInfo.getListOfToAddresses();
        for (int i = 0; i < arrayListOfToAddresses.size(); i++) {
            String toAddress = arrayListOfToAddresses.get(i);
            InternetAddress toInternetAddress = new InternetAddress(toAddress);
            mimeMessage.addRecipient(jakarta.mail.Message.RecipientType.TO, toInternetAddress);
        }
        
        MimeBodyPart mimeBodyPart = new MimeBodyPart();
        mimeBodyPart.setContent(emailInfo.getBody(), "text/plain");
        Multipart multipart = new MimeMultipart();
        multipart.addBodyPart(mimeBodyPart);
        for (DataSource attachment : emailInfo.getListOfAttachments()) {
            mimeBodyPart = new MimeBodyPart();
            mimeBodyPart.setDataHandler(new DataHandler(attachment));
            mimeBodyPart.setFileName(attachment.getName());
            multipart.addBodyPart(mimeBodyPart);
        }
        mimeMessage.setContent(multipart);
        
        // Encode and wrap the MIME message into a Gmail message
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        mimeMessage.writeTo(byteArrayOutputStream);
        byte[] arrayOfBytes = byteArrayOutputStream.toByteArray();
        String encodedEmail = Base64.getEncoder().encodeToString(arrayOfBytes);
        Message message = new Message();
        message.setRaw(encodedEmail);
        
        // See https://developers.google.com/gmail/api/quickstart/java
        // Build a new authorized API client service.
        final NetHttpTransport networkHttpTransport = GoogleNetHttpTransport.newTrustedTransport();
        JsonFactory jsonFactory = GsonFactory.getDefaultInstance();
        String fromAddress = emailInfo.getFromAddress();
        Credential credential = getCredentials(fromAddress, networkHttpTransport, jsonFactory, configurations.getCredentialsDirectory());
        Gmail.Builder gmailBuilder = new Gmail.Builder(networkHttpTransport, jsonFactory, credential);
        gmailBuilder.setApplicationName("email-sender");
        Gmail gmail = gmailBuilder.build();
        
        // See https://developers.google.com/gmail/api/guides/sending
        try {
            // Send message
            message = gmail.users().messages().send(fromAddress, message).execute();
        } catch (GoogleJsonResponseException e) {
            // TODO(developer) - handle error appropriately
            GoogleJsonError error = e.getDetails();
            if (error.getCode() == 403) {
                System.err.println("Unable to send message: " + e.getDetails());
            } else {
                throw e;
            }
        }
    }
}
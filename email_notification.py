import smtplib, ssl

def send_email(message="task completed"):
  port = 465
  context = ssl.create_default_context()

  account = "jandevacc@gmail.com"
  password = "seNvig-5hozke-kodraw"

  sender_email = "jandevacc@gmail.com"
  receiver_email = "jan.kolnik@gmail.com"
  
  with smtplib.SMTP_SSL("smtp.gmail.com", 
                        port = port, 
                        context = context) as server:
      
      login_response = server.login(user = account, 
                  password = password)
      
      # print(f"login response: {login_response}")
      send_response = server.sendmail(msg = message, 
                          from_addr=sender_email, 
                          to_addrs=receiver_email)
      if len(send_response) == 0:
        print(f"email sent")
        pass
      else:
        print(f"sending email FAILED: {send_response}")
        
      server.quit()
    
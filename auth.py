from telethon import TelegramClient

# Your credentials
API_ID = 21528285
API_HASH = '17502cc14c93332c2537b7838d6db5bf'
PHONE = '+306988738227'

async def main():
    # Create the client and connect
    client = TelegramClient('session_name', API_ID, API_HASH)
    await client.start(phone=PHONE)

    # Ensure we're signed in
    if not await client.is_user_authorized():
        await client.send_code_request(PHONE)
        code = input('Enter the code you received: ')
        await client.sign_in(PHONE, code)
        print("Successfully authenticated!")
    else:
        print("Already authenticated!")

    await client.disconnect()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
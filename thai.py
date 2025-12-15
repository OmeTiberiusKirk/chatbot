import asyncpg
import asyncio


async def get_pool():
    return await asyncpg.create_pool(
        user="postgres",
        password="P@ssw0rd",
        database="postgres",
        host="127.0.0.1",
        port=5432,
        min_size=1,
        max_size=10,
    )


async def main():
    pool = await get_pool()
    print(pool)

if __name__ == "__main__":
    asyncio.run(main())

import useSWR from 'swr'

const fetcher = (url) => fetch(url).then((res) => res.json())

const load = () => {
  const { data: data1, error: error1 } = useSWR(
    'https://dsa-ewg-live-election-results.s3.us-east-1.amazonaws.com/results/Mayor+(FAKE+DATA).json',
    fetcher,
    {
      refreshInterval: 5000,
    },
  )

  const { data: data2, error: error2 } = useSWR(
    'https://dsa-ewg-live-election-results.s3.us-east-1.amazonaws.com/results/Member+of+the+City+Council+38th+Council+District+(FAKE+DATA).json',
    fetcher,
    {
      refreshInterval: 5000,
    },
  )

  const { data: data3, error: error3 } = useSWR(
    'https://dsa-ewg-live-election-results.s3.us-east-1.amazonaws.com/results/status.json',
    fetcher,
    {
      refreshInterval: 5000,
    },
  )

  return {
    data: { mayoral: data1, council: data2, status: data3 },
    error: { mayoral: error1, council: error2, status: error3 },
  }
}

export default load
